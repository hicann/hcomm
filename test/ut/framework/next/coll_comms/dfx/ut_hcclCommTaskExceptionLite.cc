/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "hccl_comm_pub.h"
#define private public
#include "hcclCommTaskExceptionLite.h"
#include "aicpu_ts_thread.h"
#undef private
#include "hcomm_task_scheduler_error.h"
#include "aicpu_indop_env.h"
#include "adapter_hal_pub.h"
#include "dlhal_function_v2.h"
#include "hcclCommTaskException.h"
#include "rtsq_base.h"
#include "kernel_entrance.h"

using namespace hccl;
using namespace hcomm;

constexpr u32 RT_UB_LOCAL_OPERATIOINERR = 0x2;
constexpr u32 RT_UB_REMOTE_OPERATIOINERR = 0x3;
constexpr u32 RT_UB_LINK_FAILEDERR = 0x5;

class hcclCommTaskExceptionLiteTest : public testing::Test {
protected:
    virtual void SetUp() override
    {
        MOCKER(::getpid).stubs().will(returnValue(12345));
        MOCKER(HrtHalDrvQueryProcessHostPid).stubs().will(returnValue(HCCL_SUCCESS));
        Hccl::DlHalFunctionV2::GetInstance().dlHalEschedSubmitEvent
            = [](unsigned int, struct event_summary*) -> drvError_t {
            return DRV_ERROR_NONE;
        };
        HcclCommTaskExceptionLite::GetInstance().Init(0);
        g_taskExpDevMemMap.clear();
    }

    virtual void TearDown() override
    {
        g_taskExpDevMemMap.clear();
        GlobalMockObject::verify();
    }

private:
    u32 notifyId = 1;
    u32 tsId = 2;
};

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchUBCqeErrCodeToTsErrCode_When_Normal_Expect_ReturnIsCorrect)
{
    uint16_t ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_LOCAL_OPERATIOINERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_DDRC_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_REMOTE_OPERATIOINERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_POISON_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_LINK_FAILEDERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_LINK_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(static_cast<u32>(123));
    EXPECT_EQ(ret, TS_ERROR_HCCL_OTHER_ERROR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchSdmaCqeErrCodeToTsErrCode_When_Normal_Expect_ReturnIsCorrect)
{
    uint16_t ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_COMPERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_LINK_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_COMPDATAERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_POISON_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_DATAERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_DDRC_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(static_cast<u32>(123));
    EXPECT_EQ(ret, TS_ERROR_HCCL_OTHER_ERROR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchSdmaCqeErrCodeToTsErrCode_taskexception_disable)
{
    hcomm::SetTaskExceptionEnable(false);
    rtLogicCqReport_t exceptionInfo;
    dfx::CqeStatus cqeStatus = dfx::CqeStatus::kDefault;
    std::vector<std::pair<std::string, CollCommAicpuMgr*>> aicpuCommInfo;
    HcclResult ret
        = HcclCommTaskExceptionLite::GetInstance().ProcessCqe(nullptr, exceptionInfo, cqeStatus, aicpuCommInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::SetTaskExceptionEnable(true);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_UBSqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 9;
    exceptionInfo.errorCode = RT_UB_LOCAL_OPERATIOINERR;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_SDMASqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 11;
    exceptionInfo.errorCode = RT_SDMA_COMPERR;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_OtherSqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 8;
    exceptionInfo.errorCode = 123;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintAllCommTaskException)
{
    MOCKER_CPP(&CollCommAicpuMgr::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "taskException_test_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    EXPECT_EQ(AicpuIndopProcess::AicpuIndOpCommInit(&commAicpuParam), HCCL_SUCCESS);
    EXPECT_EQ(hcomm::HcclCommTaskExceptionLite::GetInstance().PrintAllCommTaskException(), HCCL_SUCCESS);
    EXPECT_EQ(AicpuIndopProcess::AicpuDestroyCommbyGroup(commAicpuParam.hcomId), HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintCommTaskException)
{
    u32 sqHead = 1;
    u32 sqTail = 2;
    MOCKER(QuerySqStatus)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(sqHead), outBound(sqTail))
        .will(returnValue(HCCL_SUCCESS));

    uint16_t streamId = 1;
    uint16_t taskId = 10;
    MOCKER_CPP(&Hccl::RtsqBase::GetStreamIdAndTaskIdBySqIdx)
        .stubs()
        .with(mockcpp::any(), outBound(streamId), outBound(taskId))
        .will(returnValue(HCCL_SUCCESS));

    CollCommAicpu aicpuComm;
    std::shared_ptr<AicpuTsThread> thread = std::make_shared<AicpuTsThread>("test");
    hccl::AicpuTsThread::HcclStreamInfo streamParam;
    streamParam.streamIds = streamId;
    streamParam.sqIds = 2;
    streamParam.cqIds = 3;
    streamParam.logicCqids = 4;
    EXPECT_EQ(thread->InitStreamLite(streamParam, 0), HCCL_SUCCESS);

    aicpuComm.threads_.push_back(thread);
    EXPECT_EQ(aicpuComm.dfx_.Init(aicpuComm.devId_, aicpuComm.identifier_, 0), HCCL_SUCCESS);
    auto dfxOpInfoOnce = std::make_shared<Hccl::DfxOpInfo>();
    aicpuComm.dfx_.mirrorTaskManagerLite_->SetCurrDfxOpInfo(dfxOpInfoOnce);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.taskPara.Notify.notifyID = 101;
    taskParam.taskPara.Notify.value = 1;

    auto taskInfo = std::make_shared<Hccl::TaskInfo>(streamId, taskId, INVALID_U64, taskParam, dfxOpInfoOnce);
    MOCKER_CPP(&Hccl::MirrorTaskManagerLite::GetTaskInfo).stubs().will(returnValue(taskInfo.get()));

    EXPECT_EQ(hcomm::HcclCommTaskExceptionLite::GetInstance().PrintCommTaskException(&aicpuComm), HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetGroupInfo_When_AicpuCommNullptr_Expect_ReturnEmpty)
{
    std::string result = HcclCommTaskExceptionLite::GetInstance().GetGroupInfo(nullptr);
    EXPECT_EQ(result, "");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetGroupInfo_When_AicpuCommValid_Expect_ReturnGroupName)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_group_name";
    std::string result = HcclCommTaskExceptionLite::GetInstance().GetGroupInfo(&aicpuComm);
    EXPECT_EQ(result, "group:[test_group_name], rankSize:[0], localRank:[0]");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_CommIdNotInMap_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_TaskexceptionVaNull_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    g_taskExpDevMemMap[testCommId] = nullptr;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_StopFlagIsOne_Expect_ClearFlagAndSetNullptr)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 1;

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(shmem[0], 0);
    EXPECT_EQ(g_taskExpDevMemMap[testCommId], nullptr);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagZero_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndDfxLiteNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest,
    Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndMirrorTaskMgrNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_ = HcclCommDfxLite();
    aicpuComm.dfx_.mirrorTaskManagerLite_.reset();
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndDfxOpInfoNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.Init(0, testCommId, 0);
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZero_Expect_SendTaskExceptionAndClearFlag)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.Init(0, testCommId, 0);
    auto dfxOpInfo = std::make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->cpuWaitAicpuNotifyId_ = 10;
    MOCKER_CPP(&Hccl::MirrorTaskManagerLite::GetCurrDfxOpInfo).stubs().will(returnValue(dfxOpInfo));
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint16_t clearedFlag = 0xFFFF;
    memcpy(&clearedFlag, shmem.data() + 1, sizeof(uint16_t));
    EXPECT_EQ(clearedFlag, 0);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_Call_ReturnHCCL_SUCCESS_When_CommStatusSuSpending)
{
    MOCKER_CPP(&CollCommAicpuMgr::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "taskException_test_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    EXPECT_EQ(AicpuIndopProcess::AicpuIndOpCommInit(&commAicpuParam), HCCL_SUCCESS);
    MOCKER_CPP(&CollCommAicpu::GetCommmStatus).stubs().will(returnValue(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING));
    hcomm::HcclCommTaskExceptionLite::GetInstance().Call();
}
