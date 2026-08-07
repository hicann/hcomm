/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "adapter_hal.h"
#include "aicpu/device/aicpu_channel_process.h"
#include "aicpu/device/dev_aicpu_ts_channel_mgr.h"
#include "channel_param.h"
#include "dlhns_function.h"
#include "externalinput_pub.h"
#include "hccl_dispatcher_ctx.h"
#include "workflow_pub.h"

using namespace hccl;

namespace {
constexpr uint32_t kKindTsRoce = 2U;

HcclResult StubHrtDrvGetLocalDevIDByHostDevID(u32 /*hostUdevid*/, u32* localDevid)
{
    if (localDevid != nullptr) {
        *localDevid = 0U;
    }
    return HCCL_SUCCESS;
}

HcclResult StubHrtDrvGetPlatformInfo(uint32_t* info)
{
    if (info != nullptr) {
        *info = 1U;
    }
    return HCCL_SUCCESS;
}
} // namespace

class AicpuChannelProcessTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenCommParamNull_Returns_PTR)
{
    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(nullptr), HCCL_E_PTR);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenChannelListNull_Returns_PTR)
{
    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = nullptr;
    res.channelDataListAddr = reinterpret_cast<void*>(0x1U);
    res.channelDataSizeListAddr = reinterpret_cast<void*>(0x1U);
    res.channelTypeListAddr = reinterpret_cast<void*>(0x1U);
    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_E_PTR);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenUnsupportedKind_Returns_NOT_SUPPORT)
{
    HcommRoceChannelRes blob{};
    blob.qpsPerConnection = 1U;
    blob.QpInfo[0].qpPtr = 0x7000ULL;

    void* dataPtr = static_cast<void*>(&blob);
    u64 sizeVal = sizeof(blob);
    u32 typeVal = 999U;
    ChannelHandle outHandle = 0ULL;

    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = static_cast<void*>(&outHandle);
    res.channelDataListAddr = static_cast<void*>(&dataPtr);
    res.channelDataSizeListAddr = static_cast<void*>(&sizeVal);
    res.channelTypeListAddr = static_cast<void*>(&typeVal);
    res.deviceInfo.deviceLogicId = 0;
    res.deviceInfo.devicePhyId = 0U;

    MOCKER(hrtSetWorkModeAicpu).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDeviceType).stubs().will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenValidTsRoceBlob_Returns_SUCCESS)
{
    MOCKER_CPP(&DlHnsFunction::DlHnsFunctionInit).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(StubHrtDrvGetPlatformInfo));
    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().will(returnValue(true));
    DevType devType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtDrvGetLocalDevIDByHostDevID).stubs().will(invoke(StubHrtDrvGetLocalDevIDByHostDevID));
    MOCKER(hrtSetWorkModeAicpu).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDeviceType).stubs().will(returnValue(HCCL_SUCCESS));

    HcommRoceChannelRes blob{};
    blob.qpsPerConnection = 1U;
    blob.localMemCount = 0U;
    blob.remoteMemCount = 0U;
    blob.localMem = nullptr;
    blob.remoteMem = nullptr;
    blob.QpInfo[0].qpPtr = 0x7000ULL;
    blob.QpInfo[0].sqIndex = 1U;
    blob.QpInfo[0].dbIndex = 2U;
    blob.localDataNotifyAddr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x8000ULL));
    blob.remoteNotifyAddr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x9000ULL));
    void* dataPtr = static_cast<void*>(&blob);
    u64 sizeVal = sizeof(blob);
    u32 typeVal = kKindTsRoce;
    ChannelHandle outHandle = 0ULL;

    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = static_cast<void*>(&outHandle);
    res.channelDataListAddr = static_cast<void*>(&dataPtr);
    res.channelDataSizeListAddr = static_cast<void*>(&sizeVal);
    res.channelTypeListAddr = static_cast<void*>(&typeVal);
    res.deviceInfo.deviceLogicId = 0;
    res.deviceInfo.devicePhyId = 0U;
    res.deviceInfo.deviceType = static_cast<u32>(DevType::DEV_TYPE_910_93);

    ASSERT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_SUCCESS);
    ASSERT_NE(outHandle, 0ULL);
    EXPECT_TRUE(DevAicpuTsChannelMgr::Instance().DestroyChannel(outHandle));
}
