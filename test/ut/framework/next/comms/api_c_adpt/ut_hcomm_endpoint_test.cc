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

class TestHcommEndpoint : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointCreate_When_EndpointNullptr_Return_HCCL_E_PTR)
{
    EndpointHandle handle;
    HcommResult ret = HcommEndpointCreate(nullptr, &handle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointCreate_When_HandleNullptr_Return_HCCL_E_PTR)
{
    EndpointDesc endpointDesc;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointDestroy_When_HandleNullptr_Return_HCCL_E_NOT_FOUND)
{
    HcommResult ret = HcommEndpointDestroy(nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointGet_When_EndpointNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommEndpointGet(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
