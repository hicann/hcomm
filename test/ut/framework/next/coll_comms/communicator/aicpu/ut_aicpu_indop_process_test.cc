/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * This SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../ut_hcomm_base.h"
#include "aicpu_indop_process.h"

class TestAicpuIndopProcess : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

// AicpuIndOpCommInit 空指针测试
TEST_F(TestAicpuIndopProcess, Ut_TestAicpuIndOpCommInit_When_CommAicpuParamNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = AicpuIndopProcess::AicpuIndOpCommInit(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// AicpuIndOpThreadInit 空指针测试
TEST_F(TestAicpuIndopProcess, Ut_TestAicpuIndOpThreadInit_When_ParamNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = AicpuIndopProcess::AicpuIndOpThreadInit(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// AicpuIndOpChannelInit 空指针测试
TEST_F(TestAicpuIndopProcess, Ut_TestAicpuIndOpChannelInit_When_CommParamNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = AicpuIndopProcess::AicpuIndOpChannelInit(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// AicpuIndOpNotifyInit 空指针测试
TEST_F(TestAicpuIndopProcess, Ut_TestAicpuIndOpNotifyInit_When_ParamNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = AicpuIndopProcess::AicpuIndOpNotifyInit(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// AicpuIndOpChannelUpdate 空指针测试
TEST_F(TestAicpuIndopProcess, Ut_TestAicpuIndOpChannelUpdate_When_ParamNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = AicpuIndopProcess::AicpuIndOpChannelUpdate(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
