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

// ======================== CCU IF Demo ========================

struct CcuIfDemoKernelArg {
    uint32_t value;
    uint64_t expected;
};

CcuResult CcuIfDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuIfDemoKernelArg*>(arg);

    ccu::Variable var;
    var = args->value;

    CCU_IF(var == args->expected)
    {
        ccu::Variable thenResult, thenAddend;
        thenAddend = 100;
        thenResult = var + thenAddend;
    }
    CCU_ELSE
    {
        ccu::Variable elseResult, elseAddend;
        elseAddend = 200;
        elseResult = var + elseAddend;
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU IF (without else) Demo ========================

struct CcuIfNoElseDemoKernelArg {
    uint32_t value;
    uint64_t threshold;
};

CcuResult CcuIfNoElseDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuIfNoElseDemoKernelArg*>(arg);

    ccu::Variable var;
    var = args->value;

    ccu::Variable result;
    result = 0;

    ccu::Variable addend;
    addend = 100;

    CCU_IF(var == args->threshold) { result = var + addend; }

    return CcuResult::CCU_SUCCESS;
}

// ======================== Nested IF: if{if{}}else{} ========================

struct CcuNestedIfOuterElseDemoKernelArg {
    uint32_t outerVal;
    uint64_t outerExpected;
    uint32_t innerVal;
    uint64_t innerExpected;
};

CcuResult CcuNestedIfOuterElseDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuNestedIfOuterElseDemoKernelArg*>(arg);

    ccu::Variable outerVar;
    outerVar = args->outerVal;

    ccu::Variable innerVar;
    innerVar = args->innerVal;

    ccu::Variable result;
    result = 0;

    ccu::Variable addend;

    CCU_IF(outerVar == args->outerExpected)
    {
        CCU_IF(innerVar == args->innerExpected)
        {
            addend = 10;
            result = result + addend;
        }
        addend = 20;
        result = result + addend;
    }
    CCU_ELSE
    {
        addend = 30;
        result = result + addend;
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== Nested IF: if{if{}else{}} ========================

struct CcuNestedIfInnerElseDemoKernelArg {
    uint32_t outerVal;
    uint64_t outerExpected;
    uint32_t innerVal;
    uint64_t innerExpected;
};

CcuResult CcuNestedIfInnerElseDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuNestedIfInnerElseDemoKernelArg*>(arg);

    ccu::Variable outerVar;
    outerVar = args->outerVal;

    ccu::Variable innerVar;
    innerVar = args->innerVal;

    ccu::Variable result;
    result = 0;

    ccu::Variable addend;

    CCU_IF(outerVar == args->outerExpected)
    {
        CCU_IF(innerVar == args->innerExpected)
        {
            addend = 10;
            result = result + addend;
        }
        CCU_ELSE
        {
            addend = 20;
            result = result + addend;
        }
        addend = 30;
        result = result + addend;
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== Nested IF: if{}if{} ========================

struct CcuNestedIfIfDemoKernelArg {
    uint32_t outerVal;
    uint64_t outerExpected;
    uint32_t innerVal;
    uint64_t innerExpected;
};

CcuResult CcuNestedIfIfDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuNestedIfIfDemoKernelArg*>(arg);

    ccu::Variable outerVar;
    outerVar = args->outerVal;

    ccu::Variable innerVar;
    innerVar = args->innerVal;

    ccu::Variable result;
    result = 0;

    ccu::Variable addend;

    CCU_IF(outerVar == args->outerExpected)
    {
        addend = 10;
        result = result + addend;
    }
    CCU_IF(innerVar == args->innerExpected)
    {
        addend = 20;
        result = result + addend;
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU WHILE Demo ========================

struct CcuWhileDemoKernelArg {
    uint32_t loopCount;
};

CcuResult CcuWhileDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuWhileDemoKernelArg*>(arg);

    ccu::Variable counter;
    counter = 0;

    ccu::Variable limit;
    limit = args->loopCount;

    ccu::Variable one;
    one = 1;

    ccu::Variable accumulator;
    accumulator = 0;

    ccu::Variable step;
    step = 10;

    CCU_WHILE(counter != args->loopCount)
    {
        accumulator = accumulator + step;
        counter = counter + one;
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU DO_WHILE_WHILE Demo (do{while{}}while) ========================

struct CcuDoWhileWhileDemoKernelArg {
    uint32_t loopCount;
};

CcuResult CcuDoWhileWhileDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuDoWhileWhileDemoKernelArg*>(arg);

    ccu::Variable counter_1;
    counter_1 = 0;
    ccu::Variable counter_2;
    counter_2 = 0;

    ccu::Variable one;
    one = 1;

    ccu::Variable accumulator;
    accumulator = 0;

    ccu::Variable step;
    step = 10;

    CCU_DO
    {
        CCU_WHILE(counter_2 != args->loopCount)
        {
            accumulator = accumulator + step;
            counter_2 = counter_2 + one;
        }
        accumulator = accumulator + step;
        counter_1 = counter_1 + one;
    }
    CCU_WHILE(counter_1 != args->loopCount);
    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU DO {} CCU_WHILE Demo (unified syntax) ========================

struct CcuDoWhileUnifiedDemoKernelArg {
    uint32_t loopCount;
};

CcuResult CcuDoWhileUnifiedDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuDoWhileUnifiedDemoKernelArg*>(arg);

    ccu::Variable counter;
    counter = 0;

    ccu::Variable one;
    one = 1;

    ccu::Variable accumulator;
    accumulator = 0;

    ccu::Variable step;
    step = 10;

    CCU_DO
    {
        accumulator = accumulator + step;
        counter = counter + one;
    }
    CCU_WHILE(counter != args->loopCount);

    return CcuResult::CCU_SUCCESS;
}

// ======================== Nested_in_IF IF: if{if{}if{}} ========================

struct CcuNestedInIfIfDemoKernelArg {
    uint32_t outerVal;
    uint64_t outerExpected;
    uint32_t innerVal_1;
    uint64_t innerExpected_1;
    uint32_t innerVal_2;
    uint64_t innerExpected_2;
};

CcuResult CcuNestedInIfIfDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuNestedInIfIfDemoKernelArg*>(arg);

    ccu::Variable outerVar;
    outerVar = args->outerVal;

    ccu::Variable innerVar_1;
    innerVar_1 = args->innerVal_1;

    ccu::Variable innerVar_2;
    innerVar_2 = args->innerVal_2;

    ccu::Variable result;
    result = 0;

    ccu::Variable addend;

    CCU_IF(outerVar == args->outerExpected)
    {
        CCU_IF(innerVar_1 == args->innerExpected_1)
        {
            addend = 10;
            result = result + addend;
        }
        CCU_IF(innerVar_2 == args->innerExpected_2)
        {
            addend = 20;
            result = result + addend;
        }
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU_IF Relational Demo (< <= > >=) ========================
// 覆盖 CcuKernel::IfBegin 中 LT/LE/GT/GE 四个新分支。四条 CCU_IF 之间独立，
// 触发 IfBegin->JumpGE/JumpGT/JumpLE/JumpLT 全部四种反向跳转 rep。

struct CcuIfRelationalDemoKernelArg {
    uint32_t value;
    uint64_t bound;
};

CcuResult CcuIfRelationalDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuIfRelationalDemoKernelArg*>(arg);

    ccu::Variable var;
    var = args->value;

    ccu::Variable result;
    result = 0;

    ccu::Variable addend;

    CCU_IF(var < args->bound)
    {
        addend = 1;
        result = result + addend;
    }
    CCU_IF(var <= args->bound)
    {
        addend = 2;
        result = result + addend;
    }
    CCU_IF(var > args->bound)
    {
        addend = 4;
        result = result + addend;
    }
    CCU_IF(var >= args->bound)
    {
        addend = 8;
        result = result + addend;
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU_IF Relational Demo (with else, GE) ========================
// CCU_IF+CCU_ELSE 单独覆盖一次 GE 分支，验证反向跳转到 elseLabel 时后续 else 块的正确性。

struct CcuIfElseRelationalDemoKernelArg {
    uint32_t value;
    uint64_t threshold;
};

CcuResult CcuIfElseRelationalDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuIfElseRelationalDemoKernelArg*>(arg);

    ccu::Variable var;
    var = args->value;

    ccu::Variable result;
    result = 0;

    ccu::Variable addend;

    CCU_IF(var >= args->threshold)
    {
        addend = 100;
        result = result + addend;
    }
    CCU_ELSE
    {
        addend = 200;
        result = result + addend;
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU_WHILE Relational Demo (LT) ========================
// 覆盖 CcuKernel::WhileBegin 中 LT 分支：反向跳转成 JumpGE 退出循环。

struct CcuWhileRelationalDemoKernelArg {
    uint32_t loopCount;
};

CcuResult CcuWhileRelationalDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuWhileRelationalDemoKernelArg*>(arg);

    ccu::Variable counter;
    counter = 0;

    ccu::Variable one;
    one = 1;

    ccu::Variable accumulator;
    accumulator = 0;

    ccu::Variable step;
    step = 10;

    CCU_WHILE(counter < args->loopCount)
    {
        accumulator = accumulator + step;
        counter = counter + one;
    }

    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU_DO ... CCU_WHILE Relational Demo (LE) ========================
// 覆盖 CcuKernel::DoWhileEnd 中 LE 分支：同向跳转成 JumpLE 回跳到 begin。

struct CcuDoWhileRelationalDemoKernelArg {
    uint32_t loopCount;
};

CcuResult CcuDoWhileRelationalDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuDoWhileRelationalDemoKernelArg*>(arg);

    ccu::Variable counter;
    counter = 0;

    ccu::Variable one;
    one = 1;

    ccu::Variable accumulator;
    accumulator = 0;

    ccu::Variable step;
    step = 10;

    CCU_DO
    {
        accumulator = accumulator + step;
        counter = counter + one;
    }
    CCU_WHILE(counter <= args->loopCount);

    return CcuResult::CCU_SUCCESS;
}

// ======================== CCU Variable-Variable Control Flow Demo ========================
struct CcuVarVarControlFlowDemoKernelArg {
    uint32_t valueA;
    uint32_t valueB;
    uint32_t loopCount;
};

CcuResult CcuVarVarControlFlowDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarVarControlFlowDemoKernelArg*>(arg);

    ccu::Variable varA;
    varA = args->valueA;

    ccu::Variable varB;
    varB = args->valueB;

    ccu::Variable result;
    result = 0;

    ccu::Variable addend;

    CCU_IF(varA == varB)
    {
        addend = 1;
        result = result + addend;
    }
    CCU_ELSE
    {
        addend = 2;
        result = result + addend;
    }

    CCU_IF(varA < varB)
    {
        addend = 4;
        result = result + addend;
    }

    CCU_IF(varA >= varB)
    {
        addend = 8;
        result = result + addend;
    }

    ccu::Variable counter;
    counter = 0;

    ccu::Variable limit;
    limit = args->loopCount;

    ccu::Variable one;
    one = 1;

    CCU_WHILE(counter < limit)
    {
        result = result + one;
        counter = counter + one;
    }

    ccu::Variable counter2;
    counter2 = 0;

    CCU_DO
    {
        result = result + one;
        counter2 = counter2 + one;
    }
    CCU_WHILE(counter2 != limit);

    return CcuResult::CCU_SUCCESS;
}
