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
#include "hcomm_adapter_hccp.h"
#include "hccp_common.h"
#include "hccp_tlv.h"

using namespace hcomm;
using namespace Hccl;

class HcommAdapterHccpTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HcommAdapterHccpTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HcommAdapterHccpTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in HcommAdapterHccpTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcommAdapterHccpTest TearDown" << std::endl;
    }
};

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_VersionEnough_Expect_Success)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));

    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);
    
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_When_VersionNotEnough_Expect_NotSupport)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION - 1;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));
    
    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);
    
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_When_RaGetInterfaceFailed_Expect_E_INTERNAL)
{
    u32 devPhyId = 1;
    MOCKER(RaGetInterfaceVersion).stubs().will(returnValue(-1));
    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, ut_HccpCheckUboeSupported_When_DevFeatureBitSet_Expect_True)
{
    u32 devFeature = 1 << UBOE_DEV_FLAG_RIGHT_SHIFT;
    bool result = HccpCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);
    
    // 测试其他位也有值的情况
    devFeature = (1 << UBOE_DEV_FLAG_RIGHT_SHIFT) | 0xFFFF;
    result = HccpCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);
}

TEST_F(HcommAdapterHccpTest, ut_HccpCheckUboeSupported_When_DevFeatureBitNotSet_Expect_False)
{
    u32 devFeature = 0;
    bool result = HccpCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);
    
    // 测试只有其他位被设置，但UBOE位未设置
    devFeature = 0xFFFFFFFF & ~(1 << UBOE_DEV_FLAG_RIGHT_SHIFT);
    result = HccpCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);
}

// ========== HccpRaTlvRequestForCustomChannel 测试 ==========

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_RaTlvRequestOk_Expect_ReturnHCCL_SUCCESS)
{
    void *tlvHandle = reinterpret_cast<void *>(0x1234);
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    MOCKER(RaTlvRequest)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_TlvHandleNull_Expect_ReturnHCCL_E_PTR)
{
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(nullptr, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_CustomInNull_Expect_ReturnHCCL_E_PTR)
{
    void *tlvHandle = reinterpret_cast<void *>(0x1234);
    char outBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, nullptr, outBuff);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_CustomOutNull_Expect_ReturnHCCL_E_PTR)
{
    void *tlvHandle = reinterpret_cast<void *>(0x1234);
    char inBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_RaTlvRequestFail_Expect_ReturnHCCL_E_NETWORK)
{
    void *tlvHandle = reinterpret_cast<void *>(0x1234);
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    MOCKER(RaTlvRequest)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(-1));

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// ========== HrtRaDumpJettyContext 测试 ==========

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_JettyHandleIsNull_Expect_ReturnPtr)
{
    HcclResult ret = HrtRaDumpJettyContext(nullptr, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_RaCtxGetJettyContextFails_Expect_ReturnInternal)
{
    void *validHandle = reinterpret_cast<void*>(0x1234);
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(-1));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_ContextLenIsZero_Expect_ReturnInternal)
{
    void *validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 0;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_ContextLenExceedsMax_Expect_ReturnInternal)
{
    void *validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 600;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_Normal_Expect_ReturnSuccess)
{
    void *validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 256;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 7);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
