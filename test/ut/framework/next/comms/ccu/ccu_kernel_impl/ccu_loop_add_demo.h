/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_primitives.hpp"
#include "ccu_types.h"

namespace ccu = ::AscendC::ccu;

struct CcuLoopAddKernelArg {
    uint32_t numA{0};
    uint32_t numB{0};
};

CcuResult CcuLoopAddDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    auto *args = static_cast<CcuLoopAddKernelArg *>(arg);

    Variable r1{}, r2{}, r3{}, r4{}, r5{}, r6{}, r7{}, numA{}, numB{};

    numA = args->numA;
    numB = args->numB;

    r1 = numA + numB;

    // ========== LoopGroup 1 (config-based): two config loops, no unroll ==========
    Func body1([&]() {
        r2 = numA + numB;
    });
    Func body2([&]() {
        r3 = numA + numB;
    });

    LoopConfig cfg1 = {.addrOffset = 0, .iterNum = 2};
    LoopConfig cfg2 = {.addrOffset = 0, .iterNum = 2};
    Loop loop1(cfg1, body1);
    Loop loop2(cfg2, body2);

    LoopGroupConfig grpCfg1 = {
        .cloneNum = 0, .cloneLoopOffset = 0,
        .addrOffset = 0, .ccuBufferOffset = 0, .eventOffset = 0
    };
    LoopGroup group1(grpCfg1, /*maxLoopNum=*/2, {loop1, loop2});

    r4 = numA + numB;

    // ========== LoopGroup 2: reuse loop2 + offset loop3 ==========
    Func body3([&]() {
        r5 = numA + numB;
    });
    LoopConfig cfg3 = {.addrOffset = 4096, .iterNum = 4};
    Loop loop3(cfg3, body3);

    LoopGroupConfig grpCfg2 = {
        .cloneNum = 3, .cloneLoopOffset = 1,
        .addrOffset = 4096, .ccuBufferOffset = 1, .eventOffset = 1
    };
    LoopGroup group2(grpCfg2, /*maxLoopNum=*/2, {loop2, loop3});

    // ========== LoopGroup 3 (var-based): variable group with two distinct var-loops ==========
    Variable varLoopParam4{}, varLoopParam5{}, varParallel{}, varOffset{};

    varLoopParam4 = 0x0001000200030000ULL;
    varLoopParam5 = 0x0002000300040000ULL;
    varParallel   = 0x0002000100020000ULL;
    varOffset     = 0x1000000100010000ULL;

    Func body4([&]() {
        r6 = numA + numB;
    });
    Func body5([&]() {
        r7 = numA + numB;
    });
    Loop loop4(varLoopParam4, body4);
    Loop loop5(varLoopParam5, body5);

    LoopGroup group3(varParallel, varOffset, /*maxLoopNum=*/2, {loop4, loop5});

    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuIfInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    Variable v{};
    v = 0;
    Func body([&]() {
        CCU_IF(v == 0) {
            Variable t{};
            t = v + v;
        }
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuNotifyRecordInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    ChannelHandle ch = 0;
    Func body([&]() {
        (void)NotifyRecord(ch, 0);
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuWriteVarWithNotifyInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    ChannelHandle ch = 0;
    Variable v{};
    v = 1;
    Func body([&]() {
        (void)WriteVariableWithNotify(ch, v, 0, 0);
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuEventRecordTagInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    Func body([&]() {
        (void)EventRecord("evt_tag", 1);
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuEventRecordInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    Event evt{};
    Func body([&]() {
        (void)EventRecord(evt, 1);
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}
