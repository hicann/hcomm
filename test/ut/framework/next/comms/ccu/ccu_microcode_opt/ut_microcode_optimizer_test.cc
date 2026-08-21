/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_microcode_v1.h"
#include "ccu_log.h"
#include "extract_operands.h"
#include "instruction_scheduler.h"
#include "microcode_optimizer.h"
#include "config/barrier_config.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace hcomm {
namespace CcuOpt {
    namespace {

        using namespace hcomm::CcuRep;

        // 按 ExtractOperandsV2 的 Def/Use 顺序重放依赖图 (lastWriter / lastReaders),
        // 收集 WAW / WAR / RAW 三类边, 供 TEST_F 直接查询 HasEdge。
        class DepGraphReplay {
        public:
            struct Edge {
                int from;
                int to;
            };

            explicit DepGraphReplay(const std::vector<CcuInstr>& seq)
            {
                for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
                    auto ops = ExtractOperandsV2(seq[i]);
                    RecordUses(ops, i);
                    RecordDefs(ops, i);
                }
            }

            bool HasEdge(char kind, int from, int to) const
            {
                const std::vector<Edge>& es = (kind == 'w') ? waw_ : (kind == 'r' ? raw_ : war_);
                for (const auto& e : es) {
                    if (e.from == from && e.to == to)
                        return true;
                }
                return false;
            }

        private:
            static uint32_t KeyOf(RegType t, uint16_t id)
            {
                return (static_cast<uint32_t>(t) << 16) | static_cast<uint32_t>(id);
            }

            void RecordUses(const std::vector<RegOperand>& ops, int i)
            {
                for (const auto& op : ops) {
                    if (op.isDef)
                        continue;
                    uint32_t k = KeyOf(op.type, op.regId);
                    auto it = lastWriter_.find(k);
                    if (it != lastWriter_.end())
                        raw_.push_back({it->second, i});
                    lastReaders_[k].push_back(i);
                }
            }

            void RecordDefs(const std::vector<RegOperand>& ops, int i)
            {
                for (const auto& op : ops) {
                    if (!op.isDef)
                        continue;
                    uint32_t k = KeyOf(op.type, op.regId);
                    AddWarEdges(k, i);
                    auto wIt = lastWriter_.find(k);
                    if (wIt != lastWriter_.end() && wIt->second != i)
                        waw_.push_back({wIt->second, i});
                    lastWriter_[k] = i;
                }
            }

            void AddWarEdges(uint32_t k, int i)
            {
                auto rIt = lastReaders_.find(k);
                if (rIt == lastReaders_.end())
                    return;
                for (int r : rIt->second) {
                    if (r != i)
                        war_.push_back({r, i});
                }
                rIt->second.clear();
            }

            std::unordered_map<uint32_t, int> lastWriter_;
            std::unordered_map<uint32_t, std::vector<int>> lastReaders_;
            std::vector<Edge> waw_;
            std::vector<Edge> war_;
            std::vector<Edge> raw_;
        };

        class MicrocodeOptTest : public ::testing::Test {
        protected:
            static CcuInstr MakeNop()
            {
                CcuInstr ins{};
                CcuV2::Nop(&ins);
                return ins;
            }
            static CcuInstr MakeLoadImdToX(uint16_t xnId, uint64_t imm)
            {
                CcuInstr ins{};
                CcuV2::LoadImdToXn(&ins, xnId, imm);
                return ins;
            }
            // Xd = Xn + Xm (parMode = 1)
            static CcuInstr MakeAdd(uint16_t xd, uint16_t xn, uint16_t xm)
            {
                CcuInstr ins{};
                CcuV2::Add(&ins, xd, xn, xm);
                return ins;
            }
            static CcuInstr MakeSetCKE(uint16_t setId, uint16_t waitId)
            {
                CcuInstr ins{};
                CcuV2::SetCKE(&ins, setId, 0xffff, waitId, 0, /*clearType*/ 1);
                return ins;
            }
            // Loop(startInstrId, endInstrIdInclusive, iterNum, offset, contextId).
            // 注意: endInstrId 是闭区间, 指向 body 的最后一条指令的全局 id (与发射器
            // ccu_ins_generater_v2.cc: `start + count - 1` 一致).
            static CcuInstr MakeLoop(uint16_t start, uint16_t endInclusive, uint16_t iterXn)
            {
                CcuInstr ins{};
                CcuV2::Loop(&ins, start, endInclusive, iterXn, 0, 0);
                return ins;
            }
            static CcuInstr MakeClearCKE(uint16_t clearId, uint16_t waitId)
            {
                CcuInstr ins{};
                CcuV2::ClearCKE(
                    &ins, clearId, /*clearMask*/ 0xffff, waitId,
                    /*waitMask*/ waitId == 0 ? uint16_t{0} : uint16_t{0xffff},
                    /*clearType*/ 1);
                return ins;
            }

            // 在 ExtractOperandsV2 结果里查找某个 CKE 寄存器是否以给定 def/use 身份出现。
            static bool HasCkeOperand(const std::vector<RegOperand>& ops, uint16_t regId, bool isDef)
            {
                for (const auto& o : ops) {
                    if (o.type == RegType::CKE && o.regId == regId && o.isDef == isDef)
                        return true;
                }
                return false;
            }
        };

        TEST_F(MicrocodeOptTest, ExtractOperands_LoadImdToX)
        {
            auto ins = MakeLoadImdToX(/*xn*/ 5, /*imm*/ 0x1234);
            auto ops = ExtractOperandsV2(ins);
            ASSERT_EQ(ops.size(), 1u);
            EXPECT_EQ(ops[0].type, RegType::XN);
            EXPECT_EQ(ops[0].regId, 5);
            EXPECT_TRUE(ops[0].isDef);
        }

        TEST_F(MicrocodeOptTest, ExtractOperands_Add)
        {
            // xd = xn + xm (parMode = 1) => def xd; use xn, xm
            auto ins = MakeAdd(/*xd*/ 10, /*xn*/ 11, /*xm*/ 12);
            auto ops = ExtractOperandsV2(ins);
            bool foundDef = false, foundXn = false, foundXm = false;
            for (const auto& o : ops) {
                if (o.type != RegType::XN)
                    continue;
                if (o.isDef && o.regId == 10)
                    foundDef = true;
                if (!o.isDef && o.regId == 11)
                    foundXn = true;
                if (!o.isDef && o.regId == 12)
                    foundXm = true;
            }
            EXPECT_TRUE(foundDef);
            EXPECT_TRUE(foundXn);
            EXPECT_TRUE(foundXm);
        }

        // setcke 的 setCKEId 只是置位供别人等待, 不产生供后续指令读取的数据, 故 **不作为 CKE def**;
        // waitCKEId 仍是阻塞读 (use). 只有 clearType=1 自动清零的 waitCKEId 才是触发写后读的 def.
        TEST_F(MicrocodeOptTest, ExtractOperands_SetCKE_SetIsNotDef_WaitIsUse)
        {
            auto ins = MakeSetCKE(/*setId*/ 7, /*waitId*/ 3);
            auto ops = ExtractOperandsV2(ins);
            bool foundSetDef = false, foundWaitUse = false;
            for (const auto& o : ops) {
                if (o.type != RegType::CKE)
                    continue;
                if (o.isDef && o.regId == 7)
                    foundSetDef = true;
                if (!o.isDef && o.regId == 3)
                    foundWaitUse = true;
            }
            EXPECT_FALSE(foundSetDef) << "setcke 的 setCKEId 只是置位, 不应作为 CKE def";
            EXPECT_TRUE(foundWaitUse) << "setcke 的 waitCKEId 仍是阻塞读 (use)";
        }

        // clearType==1: 硬件 wait 到 waitCKEId 后会自动清零, 因此 waitCKEId 既是 use 也是 def.
        TEST_F(MicrocodeOptTest, ExtractOperands_ClearCKE_ClearType1_AutoClearsWaitId)
        {
            // MakeClearCKE 内部固定 clearType=1 (与 EventWait / NotifyWait 生产路径一致).
            // 生产路径下 clearCKEId==0, waitCKEId!=0: 仅 waitCKEId 参与依赖.
            auto ins = MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5);
            auto ops = ExtractOperandsV2(ins);

            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ false)) << "waitCKEId 应作为 use (阻塞读)";
            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ true))
                << "clearType=1 下 waitCKEId 应同时作为 def (自动清零)";
            // clearCKEId==0 应被兜底丢弃, 不进入依赖图 (def/use 都不该出现)。
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 0, /*isDef*/ true));
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 0, /*isDef*/ false));
        }

        // setcke 与 clearcke 对称: clearType==1 时硬件 wait 命中后自动清零 waitCKEId, 因此 waitCKEId
        // 既是 use 也是 def (三种 wait 类 rep 的 profiling 分支即 setId=0/waitId!=0/clearType=1).
        TEST_F(MicrocodeOptTest, ExtractOperands_SetCKE_ClearType1_AutoClearsWaitId)
        {
            auto ins = MakeSetCKE(/*setId*/ 0, /*waitId*/ 5); // MakeSetCKE 默认 clearType=1
            auto ops = ExtractOperandsV2(ins);

            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ false)) << "waitCKEId 应作为 use (阻塞读)";
            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ true))
                << "clearType=1 下 waitCKEId 应同时作为 def (自动清零)";
            // setCKEId==0 应被兜底丢弃, 不进入依赖图 (def/use 都不该出现)。
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 0, /*isDef*/ true));
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 0, /*isDef*/ false));
        }

        // 反面用例: clearType==0 时 waitCKEId 只 wait 不清零, 仅作为 use; clearCKEId 只是主动清位,
        // 不作为触发写后读的 def. 因此该指令不产生任何 CKE def.
        TEST_F(MicrocodeOptTest, ExtractOperands_ClearCKE_ClearType0_NoAutoClear)
        {
            CcuInstr ins{};
            CcuV2::ClearCKE(
                &ins, /*clearId*/ 3, /*clearMask*/ 0xffff,
                /*waitId*/ 5, /*waitMask*/ 0xffff, /*clearType*/ 0);
            auto ops = ExtractOperandsV2(ins);

            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ false));
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ true)) << "clearType=0 时不应把 waitCKEId 视为 def";
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 3, /*isDef*/ true))
                << "clearCKEId 只是主动清位, 不作为触发写后读的 def";
        }

        // 建模层验证: 直接按 lastWriter / lastReaders 重放 ExtractOperandsV2 的 Def/Use.
        // 只有 clearType=1 自动清零的 waitCKEId 才是 CKE 写者: setcke(setId=1,waitId=0) 不产生 CKE def,
        // 故 CKE[1] 的写者只有两条 clearcke(自清), 写后读只在 clearcke 之间.
        TEST_F(MicrocodeOptTest, ExtractOperands_OnlySelfClearDefsCke)
        {
            // 按输入顺序构造 4 条: SetCKE(setId=1) / ClearCKE(waitId=1) / SetCKE(setId=1) / ClearCKE(waitId=1)
            std::vector<CcuInstr> seq;
            seq.push_back(MakeSetCKE(/*setId*/ 1, /*waitId*/ 0));
            seq.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 1));
            seq.push_back(MakeSetCKE(/*setId*/ 1, /*waitId*/ 0));
            seq.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 1));

            // 重放依赖图, 只关心 CKE[1] 上的 def-def 链. 'w'=WAW, 'r'=RAW.
            DepGraphReplay g(seq);

            // setcke (ins0/ins2, setId=1/waitId=0) 不产生 CKE def, 不作为任何依赖端点.
            EXPECT_FALSE(g.HasEdge('w', 0, 1)) << "setcke 非 def, 不该有 setcke->clearcke 的 WAW";
            EXPECT_FALSE(g.HasEdge('w', 1, 2)) << "ins2 是 setcke, 非 def, 不该作为 WAW 后继";
            EXPECT_FALSE(g.HasEdge('w', 2, 3)) << "ins2 是 setcke, 非 def, 不该作为 WAW 前驱";
            // CKE[1] 的真实写者只有两条 clearcke 的自清: ins1 def, ins3 def => 唯一 WAW 1->3.
            EXPECT_TRUE(g.HasEdge('w', 1, 3)) << "两条 clearcke 自清之间构成唯一 WAW: 1->3";
            // 写后读只在 clearcke 之间: ins3 读 CKE[1], 前一个写者是 ins1 => RAW 1->3.
            EXPECT_TRUE(g.HasEdge('r', 1, 3)) << "clearcke 之间的写后读: 1->3";
            // ins1 读 CKE[1] 时没有更早的写者 (setcke 非 def) => 不该有指向 ins1 的 RAW.
            EXPECT_FALSE(g.HasEdge('r', 0, 1)) << "setcke 非 def, 不该产生 setcke->clearcke 的 RAW";
        }

        TEST_F(MicrocodeOptTest, ExtractOperands_NopHasNoOperands)
        {
            auto ins = MakeNop();
            auto ops = ExtractOperandsV2(ins);
            EXPECT_TRUE(ops.empty());
        }

        // setcke(仅 setCKEId 置位) 之后紧跟读同 id 的 clearcke: setCKEId 不是 CKE def (只是置位供别人
        // 等待), 不构成写后读, 不补 NOP. 对应 "record -> 紧邻 wait 同一 event" 不该多插 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_SetCkeThenClearCke_SameBit_NoNop)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeSetCKE(/*setId*/ 5, /*waitId*/ 0));     // 置位 cke5 (非 def)
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // 读 cke5, 前无写者
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, 0u) << "setCKEId 非 def, setcke->clearcke 不构成写后读";
            EXPECT_EQ(sched.Stats().nopRemoved, 0u);
            ASSERT_EQ(out.instrVec.size(), 2u); // 顺序保持, 无插入
        }

        // clearcke(clearType=1 自动清零) 之后紧跟同 id 的 clearcke: 构成 CKE 写后读, 按
        // CCU_CKE_RAW_LATENCY 补 (L-1) 个 NOP (前一条自身占 1 cycle), 且严格 < L.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_InsertsNopsForCkeRawHazard)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // 读并清 cke5 (def)
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // 再读 cke5 (RAW)
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            int idxFirst = -1, idxSecond = -1;
            for (int i = 0; i < static_cast<int>(out.instrVec.size()); ++i) {
                const auto& ins = out.instrVec[i];
                if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::CLEARCKBIT_CODE) {
                    if (idxFirst < 0) {
                        idxFirst = i;
                    } else {
                        idxSecond = i;
                    }
                }
            }
            ASSERT_EQ(idxFirst, 0);
            EXPECT_EQ(idxSecond, static_cast<int>(CcuRep::CCU_CKE_RAW_LATENCY)); // clear(0) + (L-1) NOP + clear
            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_CKE_RAW_LATENCY - 1);
            EXPECT_LT(sched.Stats().nopInserted, CcuRep::CCU_CKE_RAW_LATENCY); // 每读者补 NOP < L, 与预留对齐
            EXPECT_EQ(sched.Stats().nopRemoved, 0u);                           // CkeOnly 不剥离
        }

        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_IgnoresXnRawHazard)
        {
            // [LoadImdToX(X1), Add(Xd=X2, X1, X5)]: XN 写后读交由硬件 interlock, CkeOnly 不补 NOP.
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeLoadImdToX(/*xn*/ 1, 1));
            input.instrVec.push_back(MakeAdd(/*xd*/ 2, /*xn*/ 1, /*xm*/ 5));
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, 0u); // XN RAW 不补
            EXPECT_EQ(sched.Stats().nopRemoved, 0u);
            ASSERT_EQ(out.instrVec.size(), 2u); // 顺序保持, 无插入
            EXPECT_EQ(out.instrVec[0].header.code, InstrCodeV2::LOADIMDTOX_CODE);
            EXPECT_EQ(out.instrVec[1].header.code, InstrCodeV2::ADD_CODE);
        }

        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_LoopRefsRemappedAfterCkeNopInsertion)
        {
            // body 内含一次 CKE 写后读 (clearcke -> 同 id clearcke), CkeOnly 在 body 内插 NOP,
            // Loop.start/end 需同步平移.
            //   [0] ClearCKE(wait cke5)  <- body[0] (def cke5)
            //   [1] ClearCKE(wait cke5)  <- body[1] (RAW)
            //   [2] LoopGroup -> loop id 3
            //   [3] Loop(start=0, end=1, iter=X20)
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5));
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5));
            {
                CcuInstr lg{};
                lg.header = CcuRep::InstrHeader(InstrCodeV2::CTRL_TYPE, InstrCodeV2::LOOPGROUP_CODE);
                lg.v2.loopGroup.startLoopInstrId = 3;
                input.instrVec.push_back(lg);
            }
            input.instrVec.push_back(MakeLoop(/*start*/ 0, /*endInclusive*/ 1, /*iter*/ 20));
            input.instrCount = 4;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            int idxLoop = -1, idxLg = -1, idxBody0 = -1, idxBody1 = -1;
            for (int i = 0; i < static_cast<int>(out.instrVec.size()); ++i) {
                const auto& ins = out.instrVec[i];
                if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::LOOP_CODE) {
                    idxLoop = i;
                }
                if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::LOOPGROUP_CODE) {
                    idxLg = i;
                }
                // 两条 clearcke: 先出现的是 body0, 后出现的是 body1.
                if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::CLEARCKBIT_CODE) {
                    if (idxBody0 < 0) {
                        idxBody0 = i;
                    } else {
                        idxBody1 = i;
                    }
                }
            }
            ASSERT_GE(idxLoop, 0);
            ASSERT_GE(idxLg, 0);
            ASSERT_EQ(idxBody0, 0);
            ASSERT_EQ(idxBody1, static_cast<int>(CcuRep::CCU_CKE_RAW_LATENCY));

            const auto& loopIns = out.instrVec[idxLoop];
            EXPECT_EQ(loopIns.v2.loop.startInstrId, static_cast<uint16_t>(idxBody0));
            EXPECT_EQ(loopIns.v2.loop.endInstrId, static_cast<uint16_t>(idxBody1));

            const auto& lgIns = out.instrVec[idxLg];
            EXPECT_EQ(lgIns.v2.loopGroup.startLoopInstrId, static_cast<uint16_t>(idxLoop));
        }

        TEST_F(MicrocodeOptTest, DefaultOptions_SchedLevelIsCkeOnly)
        {
            // 极简后端优化只有 CkeOnly 一档, 编译期恒返回 CkeOnly.
            OptimizerOptions o = MicrocodeOptimizer::DefaultOptions();
            EXPECT_EQ(o.schedLevel, SchedLevel::CkeOnly);
        }

    } // namespace
} // namespace CcuOpt
} // namespace hcomm
