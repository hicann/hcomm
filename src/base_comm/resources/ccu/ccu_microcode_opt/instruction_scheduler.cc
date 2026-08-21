/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "instruction_scheduler.h"

#include <limits>
#include <unordered_map>

#include "ccu_log.h"

#include "config/barrier_config.h" // 提供 InstrCodeV2 opcode 常量

namespace hcomm {
namespace CcuOpt {

    namespace {

        inline uint32_t RegKey(const RegOperand& operand)
        {
            return (static_cast<uint32_t>(operand.type) << 16) | static_cast<uint32_t>(operand.regId);
        }

        inline CcuRep::CcuInstr MakeNop()
        {
            // CcuInstr 为 POD (header + union, 无非平凡成员), {} 值初始化已将全部字节零化,
            // 无需再 memset_s; 仅设置 NOP 的 header 即可.
            CcuRep::CcuInstr nopInstr{};
            nopInstr.header = CcuRep::InstrHeader(InstrCodeV2::LOAD_TYPE, InstrCodeV2::NOP_CODE);
            return nopInstr;
        }

        // CkeOnly 顺序调度的可变状态: 输出序列、原->输出映射、统计信息, 以及只跟踪 CKE 写者发射
        // cycle 的表. 不做启发式放大, 保证补 NOP 有界.
        struct CkeOnlyState {
            std::vector<CcuRep::CcuInstr>& outVec;
            std::vector<int32_t>& origToOut;
            SchedulerStats& stats;
            std::unordered_map<uint32_t, int64_t> lastCkeWriterCycle{};
            int64_t cycle = 0;
        };

        inline void EmitNop(CkeOnlyState& state)
        {
            state.outVec.push_back(MakeNop());
            state.stats.originIndex.push_back(-1); // 无对应源.
            state.stats.nopInserted++;
            state.cycle++;
        }

        // 计算当前指令为满足 CKE 写后读 latency 所需的最早发射 cycle.
        inline int64_t EarliestCkeIssueCycle(const CkeOnlyState& state, const std::vector<RegOperand>& operands)
        {
            const int64_t ckeLatency = static_cast<int64_t>(CcuRep::CCU_CKE_RAW_LATENCY);
            int64_t earliest = state.cycle;
            for (const auto& operand : operands) {
                if (operand.isDef || operand.type != RegType::CKE) {
                    continue;
                }
                auto it = state.lastCkeWriterCycle.find(RegKey(operand));
                if (it == state.lastCkeWriterCycle.end()) {
                    continue;
                }
                int64_t needed = it->second + ckeLatency;
                if (needed > earliest) {
                    earliest = needed;
                }
            }
            return earliest;
        }

        // 处理单条指令: 先补齐 latency NOP, 再原序发射, 最后记录 CKE 写者的发射 cycle.
        void ScheduleOneCkeInstr(CkeOnlyState& state, const CcuRep::CcuInstr& instr, size_t originIdx)
        {
            auto operands = ExtractOperandsV2(instr);

            int64_t earliestIssueCycle = EarliestCkeIssueCycle(state, operands);
            while (state.cycle < earliestIssueCycle) {
                EmitNop(state);
            }

            state.origToOut[originIdx] = static_cast<int32_t>(state.outVec.size());
            state.outVec.push_back(instr);
            state.stats.originIndex.push_back(static_cast<int32_t>(originIdx)); // CkeOnly 顺序保持.
            int64_t issueCycle = state.cycle;
            state.cycle++;

            for (const auto& operand : operands) {
                if (!operand.isDef || operand.type != RegType::CKE) {
                    continue;
                }
                state.lastCkeWriterCycle[RegKey(operand)] = issueCycle;
            }
        }

        // 依据已确定的 out.missionStartInstrId 重新推导 missionInstrCount:
        // mission 起点落在输出序列内则取到序列尾部的长度, 否则计 0.
        inline void RecomputeMissionCount(CcuRep::CcuInstrInfo& out, uint16_t startId)
        {
            if (static_cast<uint32_t>(out.missionStartInstrId)
                < static_cast<uint32_t>(startId) + static_cast<uint32_t>(out.instrCount)) {
                out.missionInstrCount = static_cast<uint16_t>(startId + out.instrCount - out.missionStartInstrId);
            } else {
                out.missionInstrCount = 0;
            }
        }

        // 顺序扫描后处理: 修正 missionStartInstrId / missionInstrCount 与 Loop / LoopGroup 引用.
        // CkeOnly 只在原序上插入 NOP, 不重排, 故按 origToOut 平移引用即可.
        void FixReferences(
            const CcuRep::CcuInstrInfo& input, const std::vector<int32_t>& origToOut, CcuRep::CcuInstrInfo& out)
        {
            using namespace InstrCodeV2;
            const auto& origVec = input.instrVec;
            const size_t instrCount = origVec.size();
            const uint16_t startId = input.startInstrId;

            auto remapGlobal = [&](uint16_t globalId) -> uint16_t {
                // globalId 是"startId + localId" 编码的全局 id, 越界或未映射则原样返回.
                if (globalId < startId)
                    return globalId;
                uint32_t localId = static_cast<uint32_t>(globalId) - static_cast<uint32_t>(startId);
                if (localId >= origToOut.size())
                    return globalId;
                int32_t newPos = origToOut[localId];
                if (newPos < 0)
                    return globalId;
                return static_cast<uint16_t>(startId + newPos);
            };

            // missionStartInstrId 修正: 若 mission 起点原本在本序列范围内, 映射到新的位置;
            // missionInstrCount 用序列尾部长度重新推导.
            if (input.missionStartInstrId >= startId
                && static_cast<uint32_t>(input.missionStartInstrId) < static_cast<uint32_t>(startId) + instrCount) {
                out.missionStartInstrId = remapGlobal(input.missionStartInstrId);
                RecomputeMissionCount(out, startId);
            } else {
                out.missionStartInstrId = input.missionStartInstrId;
                out.missionInstrCount = input.missionInstrCount;
            }

            for (size_t originIdx = 0; originIdx < instrCount; ++originIdx) {
                const auto& origInstr = origVec[originIdx];
                int32_t outPos = origToOut[originIdx];
                if (outPos < 0)
                    continue;
                auto& outInstr = out.instrVec[outPos];
                if (origInstr.header.type == CTRL_TYPE && origInstr.header.code == LOOP_CODE) {
                    outInstr.v2.loop.startInstrId = remapGlobal(origInstr.v2.loop.startInstrId);
                    outInstr.v2.loop.endInstrId = remapGlobal(origInstr.v2.loop.endInstrId);
                } else if (origInstr.header.type == CTRL_TYPE && origInstr.header.code == LOOPGROUP_CODE) {
                    outInstr.v2.loopGroup.startLoopInstrId = remapGlobal(origInstr.v2.loopGroup.startLoopInstrId);
                }
            }
        }

    } // namespace

    CcuRep::CcuInstrInfo InstructionScheduler::Schedule(const CcuRep::CcuInstrInfo& input)
    {
        return ScheduleCkeOnly(input);
    }

    // CkeOnly 默认档: 保持原序, 只对 CKE 寄存器的写后读 (某条 setcke 写 CKE, 之后 waitcke/
    // clearcke 读同一 CKE) 按固定 cke latency 补 NOP; XN / MS 写后读交由硬件 interlock, 不补
    // 任何 NOP. 每个 CKE 读者最多补 (L-1) 条 NOP, 与"每个 wait 类 rep 预留 L 条"精确对齐.
    CcuRep::CcuInstrInfo InstructionScheduler::ScheduleCkeOnly(const CcuRep::CcuInstrInfo& input)
    {
        stats_ = {};
        const auto& origVec = input.instrVec;
        const size_t instrCount = origVec.size();

        std::vector<CcuRep::CcuInstr> outVec;
        std::vector<int32_t> origToOut;
        outVec.reserve(instrCount);
        origToOut.assign(instrCount, -1);

        // 只跟踪 CKE 写者的发射 cycle; 不做启发式放大, 保证补 NOP 有界.
        // 索引用 size_t 与 vector::size() 对齐, 避免 instrCount 逼近 65535 时 uint16_t 回绕死循环;
        // 输出条数是否越界预留区由上游 TransRepSequenceToMicrocode 按 instrVec.size() 快速失败兜底.
        CkeOnlyState state{outVec, origToOut, stats_};
        for (size_t i = 0; i < instrCount; ++i) {
            ScheduleOneCkeInstr(state, origVec[i], i);
        }

        // CkeOnly 不做 BB 切分, 用 1 作为占位 (仅统计意义).
        stats_.basicBlocks = instrCount > 0 ? 1 : 0;

        CcuRep::CcuInstrInfo out;
        out.instrVec = std::move(outVec);
        out.startInstrId = input.startInstrId;

        // instrCount 字段为 uint16_t. 正常路径下上游按 CKE 预留区申请, 优化后条数远小于 65535;
        // 但一旦补 NOP 后输出条数超过 uint16_t 上限, 直接截断会让 instrCount 与真实 instrVec 大小
        // 不一致, 进而使 FixReferences 的引用重映射错位. 此处显式记录错误再截断, 把静默数据损坏
        // 变成可观测告警; instrVec 保留完整大小, 由上游 TransRepSequenceToMicrocode 按
        // instrVec.size() > regionSize 快速失败兜底 (见 ccu_kernel_mgr.cc).
        constexpr size_t kMaxInstrCount = std::numeric_limits<uint16_t>::max();
        if (out.instrVec.size() > kMaxInstrCount) {
            HCCL_ERROR(
                "[InstructionScheduler] optimized instr count[%zu] exceeds uint16_t range[%zu]; "
                "instrCount field will be truncated, upstream region-size check will reject it.",
                out.instrVec.size(), kMaxInstrCount);
        }
        out.instrCount = static_cast<uint16_t>(out.instrVec.size());

        FixReferences(input, origToOut, out);

        return out;
    }

} // namespace CcuOpt
} // namespace hcomm
