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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "adapter_rts.h"

class CheckUbAttrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckUbAttrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckUbAttrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CheckUbAttrTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CheckUbAttrTest TearDown" << std::endl;
    }
};

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIsZero_Expect_ReturnHCCL_SUCCESS)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP;
    channelDesc.ubAttr.sqDepth = 0;

    HcommResult ret = CheckUbAttr(channelDesc, COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs16_Expect_ReturnHCCL_SUCCESS_And_AdjustTo16)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP;
    channelDesc.ubAttr.sqDepth = 16;

    HcommResult ret = CheckUbAttr(channelDesc, COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 16);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs256_Expect_ReturnHCCL_SUCCESS_And_AdjustTo256)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP;
    channelDesc.ubAttr.sqDepth = 256;

    HcommResult ret = CheckUbAttr(channelDesc, COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 256);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs15_Expect_ReturnHCCL_SUCCESS_And_AdjustTo16)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_CTP; // 设置为UB协议以触发sqDepth检查
    channelDesc.ubAttr.sqDepth = 15;

    HcommResult ret = CheckUbAttr(channelDesc, COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 16U);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs300_Expect_ReturnHCCL_SUCCESS_And_AdjustTo512)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_CTP; // 设置为UB协议以触发sqDepth检查
    channelDesc.ubAttr.sqDepth = 300;

    HcommResult ret = CheckUbAttr(channelDesc, COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 512U);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs17_Expect_ReturnHCCL_SUCCESS_And_AdjustTo32)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_CTP; // 设置为UB协议以触发sqDepth检查
    channelDesc.ubAttr.sqDepth = 17;

    HcommResult ret = CheckUbAttr(channelDesc, COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, 32);
}

TEST_F(CheckUbAttrTest, Ut_CheckUbAttr_When_SqDepthIs100_Expect_ReturnHCCL_SUCCESS_And_AdjustTo128)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_CTP; // 设置为UB协议以触发sqDepth检查

    channelDesc.ubAttr.sqDepth = 100;

    HcommResult ret = CheckUbAttr(channelDesc, COMM_ENGINE_AICPU);
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

TEST_F(
    HcommChannelDescInitTest,
    Ut_HcommChannelDescInitAndCheckUbAttr_When_SqDepthIsNotConfigured_Expect_RemainsNotConfigured)
{
    HcommChannelDesc channelDesc{};

    HcommResult ret = HcommChannelDescInit(&channelDesc, 1);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET);

    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP;
    ret = CheckUbAttr(channelDesc, COMM_ENGINE_AIV);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubAttr.sqDepth, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET);
}

// 直调 HcommChannelCreate 归一化路径：HcommChannelDesc.qos → roceAttr.sl/tc
TEST_F(CheckUbAttrTest, Ut_CheckRoceAttr_When_QosUnset_KeepOriginalSlTc)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.roceAttr.sl = 3;
    channelDesc.roceAttr.tc = 120;
    channelDesc.qos = 0xFFFFFFFFU;

    ASSERT_EQ(CheckRoceAttr(channelDesc), HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.roceAttr.sl, 3);
    EXPECT_EQ(channelDesc.roceAttr.tc, 120);
}

TEST_F(CheckUbAttrTest, Ut_CheckRoceAttr_When_QosSet_MapsSlAndDefaultDscpTc)
{
    // ApplyRoceQosCompatToSlTc 仅 950/960 生效，需 mock 设备类型
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.roceAttr.sl = 3;
    channelDesc.roceAttr.tc = 120;
    channelDesc.qos = 4; // 未 mock HCCN 时回退默认 DSCP=33，TC=33<<2

    ASSERT_EQ(CheckRoceAttr(channelDesc), HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.roceAttr.sl, 4);
    EXPECT_EQ(channelDesc.roceAttr.tc, static_cast<uint8_t>(33U << 2U));
}

TEST_F(CheckUbAttrTest, Ut_CheckRoceAttr_When_DeviceTypeNot950Or960_Expect_SkipQosCompat)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));

    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.roceAttr.sl = 3;
    channelDesc.roceAttr.tc = 120;
    channelDesc.qos = 4;

    ASSERT_EQ(CheckRoceAttr(channelDesc), HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.roceAttr.sl, 3);
    EXPECT_EQ(channelDesc.roceAttr.tc, 120);
}

TEST_F(CheckUbAttrTest, Ut_CheckRoceAttr_When_DeviceTypeInvalid_Expect_ReturnHCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_COUNT)).will(returnValue(HCCL_SUCCESS));

    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.qos = 4;

    EXPECT_EQ(CheckRoceAttr(channelDesc), HCCL_E_PARA);
}
