/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "hccl/hccl_res.h"
#include "../../hccl_api_base_test.h"
#include "hccl_tbe_task.h"
#include "thread_manager.h"
#include "launch_aicpu.h"
#include "aicpu_launch_manager.h"

using namespace hccl;

class ThreadManagerTest : public BaseInit {
public:
    void SetUp() override
    {
        std::cout << "ThreadManagerTest SetUp" << std::endl;
        BaseInit::SetUp();
        MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
        ManagerCallbacks callbacks;
        callbacks.getAicpuCommState = []() {
            return true;
        };
        callbacks.setAicpuCommState = [](bool) {};
        callbacks.kernelLaunchAicpuCommInit = []() {
            return HCCL_SUCCESS;
        };
        callbacks.reportProfilingKernel = [](uint64_t, std::string) {
            return HCCL_SUCCESS;
        };
        threadManager = std::make_unique<ThreadMgr>(1, 1, "test", nullptr, callbacks);
    }
    void TearDown() override
    {
        std::cout << "ThreadManagerTest TearDown" << std::endl;
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

private:
    std::unique_ptr<ThreadMgr> threadManager;
    uint32_t threadNum = 1;
    ThreadHandle threads[1] = {0};
    ThreadHandle exportedThreads[1] = {0};
};

void MockGetRunSideIsDevice()
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
}

void MockThreadKernelLaunchForComm()
{
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(HCCL_SUCCESS));
}

TEST_F(ThreadManagerTest, Ut_ThreadExportToCommEngineAicpu_When_InvalidThreadHandle_Expect_HCCL_E_PARA)
{
    CommEngine dstCommEngine = COMM_ENGINE_AICPU_TS;

    HcclResult ret = threadManager->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ThreadManagerTest, Ut_ThreadExportToCommEngineAicpu_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MockGetRunSideIsDevice();
    MockThreadKernelLaunchForComm();

    CommEngine dstCommEngine = COMM_ENGINE_AICPU_TS;

    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    if (ret == HCCL_SUCCESS) {
        ret = threadManager->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}
