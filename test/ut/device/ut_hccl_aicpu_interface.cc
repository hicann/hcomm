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
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif

#include "hccl_aicpu_interface.h"
#include "aicpu_communicator.h"
#include "aicpu_hccl_process.h"
#include "aicpu_sqe_context.h"
#include "aicpu_hdc_utils.h"

#undef private
#undef protected

using namespace std;
using namespace hccl;

class Test_Hccl_Aicpu_Interface : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UT_Hccl_Aicpu_Interface SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "UT_Hccl_Aicpu_Interface TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

HcclResult GetSuspendingFlagStub(HcclCommAicpu* This, HcclComSuspendingFlag& flag)
{
    flag = HcclComSuspendingFlag::isSuspending;
    return HCCL_SUCCESS;
}

HcclCommAicpu* CreateHcclCommAicpuWithDefaultDfx()
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu();
    DfxExtendInfo* dfxInfo = hcclCommAicpu->GetDfxExtendInfo();
    dfxInfo->cqeStatus = dfx::CqeStatus::kDefault;
    dfxInfo->pollStatus = PollStatus::kDefault;
    return hcclCommAicpu;
}

HcclOpResParam* CreateHcclOpResParam(const char* groupName)
{
    HcclOpResParam* commParam = new HcclOpResParam();
    memset(commParam, 0, sizeof(HcclOpResParam));
    strcpy(commParam->hcomId, groupName);
    return commParam;
}

OpTilingData* CreateOpTilingData(HcclCMDType opType, const char* tag)
{
    OpTilingData* tilingData = new OpTilingData();
    memset(tilingData, 0, sizeof(OpTilingData));
    tilingData->opType = static_cast<u8>(opType);
    strcpy(tilingData->tag, tag);
    tilingData->workflowMode = 0;
    tilingData->isZeroCopy = 0;
    tilingData->isSymmetricMemory = 0;
    return tilingData;
}

void* BuildKfcTaskComm(HcclOpResParam* commParam, OpTilingData* tilingData)
{
    u64 taskSize = sizeof(KFCTaskComm);
    u64 totalSize = taskSize + sizeof(OpTilingData);
    u8* buffer = new u8[totalSize];
    memset(buffer, 0, totalSize);

    KFCTaskComm* task = reinterpret_cast<KFCTaskComm*>(buffer);
    task->context = reinterpret_cast<u64>(commParam);

    OpTilingData* tilingInTask = reinterpret_cast<OpTilingData*>(buffer + sizeof(KFCTaskComm));
    memcpy(tilingInTask, tilingData, sizeof(OpTilingData));
    task->tilingData = reinterpret_cast<u64>(tilingInTask);

    return reinterpret_cast<void*>(task);
}

void SetupRpcSrvLaunchMocks(HcclCommAicpu* hcclCommAicpu)
{
    MOCKER(AicpuHcclProcess::AicpuGetCommbyGroup).stubs().will(returnValue(hcclCommAicpu));

    MOCKER_CPP(&HcclCommAicpu::GetSuspendingFlag).stubs().will(invoke(GetSuspendingFlagStub));

    MOCKER(AicpuHcclProcess::AicpuReleaseCommbyGroup).stubs().will(ignoreReturnValue());
}

TEST_F(Test_Hccl_Aicpu_Interface, Ut_RunAicpuRpcSrvLaunchV2_When_SuspendingFlagIsSuspending_Expect_ReturnZero)
{
    HcclCommAicpu* hcclCommAicpu = CreateHcclCommAicpuWithDefaultDfx();
    SetupRpcSrvLaunchMocks(hcclCommAicpu);

    HcclOpResParam* commParam = CreateHcclOpResParam("test_group");
    OpTilingData* tilingData = CreateOpTilingData(HcclCMDType::HCCL_CMD_ALL, "test_tag");
    void* task = BuildKfcTaskComm(commParam, tilingData);

    uint32_t ret = RunAicpuRpcSrvLaunchV2(task);

    EXPECT_EQ(ret, 0);

    u8* buffer = reinterpret_cast<u8*>(task);
    delete[] buffer;
    delete commParam;
    delete tilingData;
    delete hcclCommAicpu;
}
