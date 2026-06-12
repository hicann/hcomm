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
#include "hcomm_c_adpt.h"
#include "local_notify_impl.h"
#include "aicpu_launch_manager.h"
#include "llt_hccl_stub_rank_graph.h"
#include "launch_aicpu.h"

class TestHcclThread : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        unsetenv("HCCL_DFS_CONFIG");
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateHostCpuTsCommEngineThread_Return_HCCL_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    std::shared_ptr<Thread> cpuHandle;
    HcclResult ret
        = CreateThread(COMM_ENGINE_CPU_TS, StreamType::STREAM_TYPE_ONLINE, 3, NotifyLoadType::HOST_NOTIFY, cpuHandle);
    EXPECT_EQ(ret, 0);
    ret = cpuHandle->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint64_t cpu = reinterpret_cast<ThreadHandle>(cpuHandle.get());
    Stream *stream = GetStream(cpu);
    EXPECT_NE(stream, nullptr);
    void *notify = GetNotify(cpu, 0);
    EXPECT_NE(nullptr, notify);
}

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateAicpuTsCommEngineThread_Return_HCCL_Success)
{
    std::shared_ptr<Thread> aicpuHandle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = CreateThread(
        COMM_ENGINE_AICPU_TS, StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY, aicpuHandle);
    EXPECT_EQ(ret, 0);
    ret = aicpuHandle->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint64_t aicpu = reinterpret_cast<ThreadHandle>(aicpuHandle.get());
    Stream *aicpuStream = GetStream(aicpu);
    EXPECT_NE(aicpuStream, nullptr);
    void *aicpuNotify = GetNotify(aicpu, 0);
    EXPECT_NE(nullptr, aicpuNotify);
}

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateNotSurportCommEngineThread_Return_HCCL_E_NOT_SUPPORT)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret
        = CreateThread(COMM_ENGINE_AIV, StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY, Handle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestHcclThread, UT_When_DeviceSide_ResourceAllocateFail_expect_return_HcclEInternal)
{
    bool isDeviceSide{true};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_ThreadIsNullptr_Allocate_expect_Return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    uint64_t *thread = nullptr;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_WithUnsupportedEngine_expect_return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AIV, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_Thread_Allocate_0Num_expect_Return_HCCL_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AIV, 0, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_WithNotifyInitFail_expect_return_HcclERuntime)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtNotifyGetOffset).stubs().will(returnValue(HCCL_E_RUNTIME));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_CpuTsThread_Allocate_expect_Return_HCCL_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Thread *threadptr0 = reinterpret_cast<Thread *>(thread[0]);
    Thread *threadptr1 = reinterpret_cast<Thread *>(thread[1]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);
    EXPECT_EQ(threadptr1->GetNotifyNum(), 3);
    ret = HcommThreadFree(thread, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_Allocate_MAXThreadNum_expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    HcommResult ret
        = HcommThreadAlloc(COMM_ENGINE_CPU_TS, hccl::HCOMM_THREADNUM_MAX_NUM + 1, static_cast<uint32_t>(0), &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_Allocate_MaxNotifyNum_expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, hccl::HCOMM_NOTIFY_MAX_NUM + 1, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_expect_Return_HCCL_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[2];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Thread *threadptr0 = reinterpret_cast<Thread *>(thread[0]);
    Thread *threadptr1 = reinterpret_cast<Thread *>(thread[1]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);
    EXPECT_EQ(threadptr1->GetNotifyNum(), 3);
    ret = HcommThreadFree(thread, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_ThreadNum_Is_0_expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[2];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Thread *threadptr0 = reinterpret_cast<Thread *>(thread[0]);
    Thread *threadptr1 = reinterpret_cast<Thread *>(thread[1]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);
    EXPECT_EQ(threadptr1->GetNotifyNum(), 3);
    ret = HcommThreadFree(thread, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_ThreadNullptr_expect_Return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle *thread = nullptr;
    HcommResult ret = HcommThreadFree(thread, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_Allocate_WithStream_expect_return_HcclSuccess)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_ThreadNullptr_expect_return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();
    ThreadHandle *thread{nullptr};
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_WithInvalidEngine_expect_return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_AICPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_NotifyInitFailed_expect_return_HCCL_E_RUNTIME)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtNotifyGetOffset).stubs().will(returnValue(HCCL_E_RUNTIME));
    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_CpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_AicpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommProfiling::ReportKernel).stubs().will(returnValue(0));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
}
TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_AicpuTsThread_Reuse_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&AicpuLaunchMgr::SupplementNotifyKernelLaunch).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommProfiling::ReportKernel).stubs().will(returnValue(0));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 2, 3, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 3, 3, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 3, 4, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 4, 5, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));

    void *comm = nullptr;
    ThreadHandle thread;
    HcclResult ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_ThreadNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle *thread{nullptr};
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_CollCommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(true));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_Normal_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread *threadptr0 = reinterpret_cast<Thread *>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(notifyNum, 3);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_Normal_Reuse_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuLaunchMgr::SupplementNotifyKernelLaunch).stubs().will(returnValue(0));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1; // 非CCU模式，避免拉起CCU平台层
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);
    Thread *threadptr0 = reinterpret_cast<Thread *>(thread[0]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 3, 3, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread *>(thread[1]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 3, 4, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread *>(thread[2]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 4);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 4, 5, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread *>(thread[3]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 5);

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(notifyNum, 5);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread *threadptr0 = reinterpret_cast<Thread *>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t notifyNum;
    comm = nullptr;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_notifyNumNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread *threadptr0 = reinterpret_cast<Thread *>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t *notifyNum{nullptr};
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_CollCommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread[2];
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());

    uint32_t notifyNum;
    HcclResult ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_Acquire_CpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();
    ThreadHandle thread;
    void *comm = nullptr;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_threadNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();
    ThreadHandle *thread{nullptr};
    void *comm = (void *)0x1234;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_rtStreamNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void *rtStream = nullptr;
    ThreadHandle thread;
    void *comm = (void *)0x1234;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_CollcommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));
    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_A3_Acquire_CpuTsThread_Return_HCCL_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireWithStream).stubs().will(returnValue(HCCL_SUCCESS));

    Stream *stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void *rtStream = stream->ptr();

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_MAX_Thread_AicpuTsThread_Return_HCCL_E_UNAVAIL)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 201, 0, &thread);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_MAX_Notify_AicpuTsThread_Return_HCCL_E_UNAVAIL)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 65537, &thread);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_Normal_Expect_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);

    ThreadResTypeStream stream;
    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_commNull_Return_HCCL_E_PTR)
{
    HcclComm comm = nullptr;
    ThreadHandle thread;
    ThreadResTypeStream stream;
    HcclResult ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_infoNull_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_threadNotFound_Return_HCCL_E_NOT_FOUND)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_restypeNotSupport_Return_HCCL_E_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);

    ThreadResTypeStream stream;
    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_INVALID, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_infoLenMismatch_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, 1, &stream);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_ApplyRdmaTrafficAndServiceLevel_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;    // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 120; // 配置RDMA Traffic Class为120，验证正常配置RDMA Traffic Class的场景
    config.hcclRdmaServiceLevel = 3;   // 配置RDMA Service Level为3，验证正常配置RDMA Service Level的场景
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_RdmaServiceLevel_Abnormal_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 8;          // 配置RDMA Service Level超过正常值
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_RdmaTrafficClass_Abnormal_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 300;        // 配置RDMA Traffic Class超过约束范围
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestHcclThread, Ut_ThreadKernelLaunchDestroy)
{
    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

    constexpr u32 threadHandleNum = 1;
    vector<ThreadHandle> deviceHandle(threadHandleNum);
    void *binHandle = nullptr;
    HcclResult ret = AicpuLaunchMgr::ThreadKernelLaunchDestroy(deviceHandle.data(), deviceHandle.size(), binHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TEST_F(TestHcclThread, Ut_LaunchNotifyKernel)
// {
//     MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
//     MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
//     MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

//     NotifyMgrAicpuParam opParam;
//     void* binCustomHandle = nullptr;
//     HcclResult ret = AicpuLaunchMgr::LaunchNotifyKernel(opParam, binCustomHandle);
//     EXPECT_EQ(ret, HCCL_SUCCESS);
// }