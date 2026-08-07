/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"

class OpBaseMiscTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();

        UT_USE_1SERVER_1RANK_AS_DEFAULT;
        // 将enableEntryLog默认返回为true
        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    }

    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(OpBaseMiscTest, Ut_HcclConfigGetInfo_When_CollCommIsNotInit_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    void* info = nullptr;
    HcclOpExpansionMode opExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    HcclResult ret = HcclConfigGetInfo(
        comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, sizeof(HcclOpExpansionMode), &opExpansionMode);
    EXPECT_EQ(ret, HCCL_E_PTR);
    Ut_Comm_Destroy(comm);
}

TEST_F(OpBaseMiscTest, Ut_HcclCommSymWinGet_When_GetCommSymWinSucceeds_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    MOCKER_CPP(&hcclComm::GetCommSymWin)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    u8 buffer = 0;
    HcclCommSymWindow winHandle = nullptr;
    size_t offset = 0;
    HcclResult ret = HcclCommSymWinGet(comm, &buffer, sizeof(buffer), &winHandle, &offset);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    Ut_Comm_Destroy(comm);
}
