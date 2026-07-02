/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"

class TestHcommMem : public TestHcommCAdptBase {
public:
    void SetUp() override {
        TestHcommCAdptBase::SetUp();
    }
    void TearDown() override {
        TestHcommCAdptBase::TearDown();
    }
};

TEST_F(TestHcommMem, Ut_TestHcommMemReg_When_InvalidHandle_Return_HCCL_E_NOT_FOUND)
{
    CommMem mem;
    mem.addr = malloc(1024);
    mem.size = 1024;
    mem.type = COMM_MEM_TYPE_HOST;
    void* memHandle = nullptr;

    EndpointHandle invalidHandle = reinterpret_cast<EndpointHandle>(0x0FFFFFFFFFFFFFFF);
    HcommResult ret = HcommMemReg(invalidHandle, "test_mem", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    free(mem.addr);
}

TEST_F(TestHcommMem, Ut_TestHcommMemReg_When_MemHandleNullptr_Return_HCCL_E_PTR)
{
    CommMem mem;
    mem.addr = malloc(1024);
    mem.size = 1024;
    mem.type = COMM_MEM_TYPE_HOST;

    HcommResult ret = HcommMemReg(nullptr, "test_mem", &mem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    free(mem.addr);
}

TEST_F(TestHcommMem, Ut_TestHcommMemUnreg_When_InvalidHandle_Return_HCCL_E_PTR)
{
    EndpointHandle invalidHandle = reinterpret_cast<EndpointHandle>(0xFFFFFFFFFFFFFFFF);
    HcommResult ret = HcommMemUnreg(invalidHandle, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemExport_When_InvalidMemHandle_Return_HCCL_E_PTR)
{
    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    HcommResult ret = HcommMemExport(nullptr, nullptr, &memDesc, &memDescLen);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemExport_When_OutputNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommMemExport(nullptr, this, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemImport_When_InvalidDesc_Return_HCCL_E_PTR)
{
    HcommMem outMem;
    HcommResult ret = HcommMemImport(nullptr, nullptr, 0, &outMem);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemUnimport_When_InvalidParams_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommMemUnimport(nullptr, nullptr, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemGetAllMemHandles_When_Nullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommMemGetAllMemHandles(nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
