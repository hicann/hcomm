/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_MICROCODE_OPT_INSTRUCTION_SCHEDULER_H
#define CCU_MICROCODE_OPT_INSTRUCTION_SCHEDULER_H

#include <cstdint>
#include <vector>

#include "ccu_instr_info_v1.h"
#include "extract_operands.h"

namespace hcomm {
namespace CcuOpt {

    struct SchedulerStats {
        uint32_t nopRemoved = 0;     // 输入序列里被剥离的 NOP 数 (CkeOnly 不剥离, 恒 0)
        uint32_t nopInserted = 0;    // 调度过程因 latency 阻塞而新插入的 NOP 数
        uint32_t instrReordered = 0; // 与输入顺序不同的真实指令数 (CkeOnly 不重排, 恒 0)
        uint32_t basicBlocks = 0;    // 切分得到的 BB 个数 (CkeOnly 不切分, 占位为 1)

        // originIndex[i] = 调度后第 i 条指令对应的输入本地下标; -1 表示新插入的填充 NOP.
        std::vector<int32_t> originIndex{};
        std::vector<uint16_t> strippedNopIndices{}; // CkeOnly 不剥离, 恒为空.
    };

    // 指令调度算法档位. 极简后端优化只提供 CkeOnly 一档:
    //  * CkeOnly: 默认档 (唯一档). 完全不重排 / 不剥离, 只识别 CKE 寄存器的写后读 (setcke ->
    //             waitcke/clearcke) 并按固定 cke latency 补 NOP; XN / MS 写后读交由硬件
    //             interlock 处理, 不补任何 NOP. 配合指令空间预留可保证优化后指令数不越界.
    enum class SchedLevel : uint8_t {
        CkeOnly = 0,
    };

    struct InstructionSchedulerOptions {
        SchedLevel level = SchedLevel::CkeOnly;
    };

    class InstructionScheduler {
    public:
        explicit InstructionScheduler(InstructionSchedulerOptions opts = {}) : opts_(opts) {}

        CcuRep::CcuInstrInfo Schedule(const CcuRep::CcuInstrInfo& input);

        const SchedulerStats& Stats() const { return stats_; }

    private:
        InstructionSchedulerOptions opts_;
        SchedulerStats stats_{};

        // CkeOnly (默认档) 实现: 仅顺序扫描 + 只对 CKE 写后读按固定 cke latency 补 NOP,
        // XN / MS 写后读不补 (硬件 interlock).
        CcuRep::CcuInstrInfo ScheduleCkeOnly(const CcuRep::CcuInstrInfo& input);
    };

} // namespace CcuOpt
} // namespace hcomm

#endif // CCU_MICROCODE_OPT_INSTRUCTION_SCHEDULER_H
