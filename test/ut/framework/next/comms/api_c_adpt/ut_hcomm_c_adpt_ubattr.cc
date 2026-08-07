/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"

class CheckUbAttrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckUbAttrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckUbAttrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CheckUbAttrTest SetUp" << std::endl; }

    virtual void TearDown() { std::cout << "A Test case in CheckUbAttrTest TearDown" << std::endl; }
};

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIsZero_Expect_ReturnHCCL_SUCCESS)
{
    HcommChannelDesc channelDesc{};
    channelDesc.ubAttr.sqDepth = 0;

    HcommResult ret = CheckUbAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs16_Expect_ReturnHCCL_SUCCESS_And_AdjustTo16)
{
    HcommChannelDesc channelDesc{};
    channelDesc.ubAttr.sqDepth = 16;

    HcommResult ret = CheckUbAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 16);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs256_Expect_ReturnHCCL_SUCCESS_And_AdjustTo256)
{
    HcommChannelDesc channelDesc{};
    channelDesc.ubAttr.sqDepth = 256;

    HcommResult ret = CheckUbAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 256);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs15_Expect_ReturnHCCL_E_PARA)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP; // 设置为UB协议以触发sqDepth检查
    channelDesc.ubAttr.sqDepth = 15;

    HcommResult ret = CheckUbAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs300_Expect_ReturnHCCL_E_PARA)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP; // 设置为UB协议以触发sqDepth检查
    channelDesc.ubAttr.sqDepth = 300;

    HcommResult ret = CheckUbAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs17_Expect_ReturnHCCL_SUCCESS_And_AdjustTo32)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP; // 设置为UB协议以触发sqDepth检查
    channelDesc.ubAttr.sqDepth = 17;

    HcommResult ret = CheckUbAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 32);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs100_Expect_ReturnHCCL_SUCCESS_And_AdjustTo128)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP; // 设置为UB协议以触发sqDepth检查

    channelDesc.ubAttr.sqDepth = 100;

    HcommResult ret = CheckUbAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 128);
}

class HcommChannelDescInitTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcommChannelDescInitTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcommChannelDescInitTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcommChannelDescInitTest SetUp" << std::endl; }

    virtual void TearDown() { std::cout << "A Test case in HcommChannelDescInitTest TearDown" << std::endl; }
};

TEST_F(HcommChannelDescInitTest, Ut_HcommChannelDescInit_When_Normal_Expect_SqDepthIsZero)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP; // 设置为UB协议以触发sqDepth检查
    channelDesc.ubAttr.sqDepth = 0xFFFFFFFF;

    HcommResult ret = HcommChannelDescInit(&channelDesc, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 0xFFFFFFFF);
}
