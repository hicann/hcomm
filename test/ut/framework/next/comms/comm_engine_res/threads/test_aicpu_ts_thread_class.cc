/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../hccl_api_base_test.h"
#include "local_notify_impl.h"
#include "llt_hccl_stub_rank_graph.h"
class TestAicpuTsThread : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_A3_HostWhen_Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Stream* stream = aicpuThread.GetStream();
    EXPECT_NE(stream, nullptr);
    uint32_t notifyNum = aicpuThread.GetNotifyNum();
    EXPECT_EQ(2, notifyNum);
    void* notify = aicpuThread.GetNotify(1);
    EXPECT_NE(nullptr, notify);
}

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_A5_Host_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Stream* stream = aicpuThread.GetStream();
    EXPECT_NE(stream, nullptr);
    uint32_t notifyNum = aicpuThread.GetNotifyNum();
    // thread内部暂时会多申请一个notify用于host&device侧同步
    EXPECT_EQ(3, notifyNum);
    void* notify = aicpuThread.GetNotify(1);
    EXPECT_NE(nullptr, notify);
}

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_A3_Device_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string mainStr = aicpuThread.GetUniqueId();
    isDeviceSide = true;
    GlobalMockObject::verify();
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread mainDevThread(mainStr);
    ret = mainDevThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_Device_When_ParamsAreForHost_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string mainStr = aicpuThread.GetUniqueId();
    isDeviceSide = true;
    GlobalMockObject::verify();
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread mainDevThread(mainStr);
    ret = mainDevThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_Device_When_GetLocalDevIDByHostDevIDFailed_Expect_Return_HCCL_E_DRV)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string mainStr = aicpuThread.GetUniqueId();
    isDeviceSide = true;
    GlobalMockObject::verify();
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtDrvGetLocalDevIDByHostDevID).stubs().will(returnValue(HCCL_E_DRV));

    AicpuTsThread mainDevThread(mainStr);
    ret = mainDevThread.Init();
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_A5_Device_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string mainStr = aicpuThread.GetUniqueId();
    isDeviceSide = true;
    GlobalMockObject::verify();
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread mainDevThread(mainStr);
    ret = mainDevThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_Host_When_hrtNotifyGetOffsetFailed_Expect_Return_HCCL_E_RUNTIME)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyGetOffset).stubs().will(returnValue(HCCL_E_RUNTIME));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_A5_Device_When_IsNormal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string mainStr = aicpuThread.GetUniqueId();
    isDeviceSide = true;
    GlobalMockObject::verify();
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread mainDevThread(mainStr);
    ret = mainDevThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* expectPtr = reinterpret_cast<void*>(0x2345);
    void* streamPtr = mainDevThread.GetStreamLitePtr();
    EXPECT_NE(nullptr, streamPtr);

    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(streamPtr);
    EXPECT_NE(nullptr, streamLite);
}

TEST_F(TestAicpuTsThread, Ut_AicpuTsThread_Init_On_A5_Device_Get_StreamId_Notify_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string mainStr = aicpuThread.GetUniqueId();
    GlobalMockObject::verify();
    u32 notifyNum = 0;
    std::string notifyDesc;
    ret = aicpuThread.GetNotifyByUniqueId(notifyNum, notifyDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    AicpuTsThread mainDevThread(mainStr);
    ret = mainDevThread.SupplementNotify(notifyNum, notifyDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestAicpuTsThread, Ut_GetNotify_When_IndexOutOfRange_Expect_Return_Nullptr)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    LocalNotify* notify = aicpuThread.GetNotify(5);
    EXPECT_EQ(nullptr, notify);
}
