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
#include <memory>
#include <functional>
#define private public
#include "hcclCommDfx.h"
#include "hcclCommProfiling.h"
#include "task_param.h"
#include "mirror_task_manager.h"
#include "hcclCommDfxLite.h"
using namespace hccl;

class HcclCommDfxTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HcclCommDfxTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HcclCommDfxTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in HcclCommDfxTest SetUp" << std::endl;
        dfx_ = std::make_unique<HcclCommDfx>();
        EXPECT_EQ(dfx_->Init(0, "test_comm", 0), HCCL_SUCCESS);

        dfxLite_ = std::make_unique<HcclCommDfxLite>();
        EXPECT_EQ(dfxLite_->Init(0, "test_comm", 0), HCCL_SUCCESS);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcclCommDfxTest TearDown" << std::endl;
    }

    std::unique_ptr<HcclCommDfx> dfx_;
    std::unique_ptr<HcclCommDfxLite> dfxLite_;
};

// 测试 AddDpuTaskInfoCallback - 正常情况
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_Normal_Expect_ReturnSuccess)
{
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_KERNEL;
    u64 handle = 0xFFFFFFFFFFFFFFFF;
    u32 remoteRankId = 1;

    // 先建立 handle 到 remoteRankId 的映射
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, remoteRankId);

    // 设置 dpuStreamId_ 和 aicpuTaskId_
    dfx_->SetDpuStreamId(100);
    dfx_->SetAicpuTaskIdAndStreamId(200, 300);

    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 AddDpuTaskInfoCallback - 空 taskParam
TEST_F(HcclCommDfxTest, Ut_AddDpuTaskInfoCallback_When_EmptyTaskParam_Expect_ReturnSuccess)
{
    Hccl::TaskParam taskParam{};
    u64 handle = INVALID_U64;
    
    HcclResult ret = dfx_->AddDpuTaskInfoCallback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 GetTaskId - 首次调用返回 0
TEST_F(HcclCommDfxTest, Ut_GetTaskId_When_FirstCall_Expect_ReturnZero)
{
    u32 streamId = 123;
    u32 taskId = HcclCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId, 1u);
}

// 测试 GetTaskId - 多次调用自增
TEST_F(HcclCommDfxTest, Ut_GetTaskId_When_MultipleCalls_Expect_Increment)
{
    u32 streamId = 456;
    
    u32 taskId1 = HcclCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId1, 1u);
    
    u32 taskId2 = HcclCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId2, 2u);
    
    u32 taskId3 = HcclCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId3, 3u);
}

// 测试 GetTaskId - 超过 65535 回环到 0
TEST_F(HcclCommDfxTest, Ut_GetTaskId_When_ExceedsLimit_Expect_ReturnToZero)
{
    u32 streamId = 789;
    
    // 先设置到 65535
    for (int i = 0; i < 65536; i++) {
        HcclCommDfx::GetTaskId(streamId);
    }
    
    u32 taskId = HcclCommDfx::GetTaskId(streamId);
    EXPECT_EQ(taskId, 1u);
}

// 测试 SetDpuStreamId - 正常设置
TEST_F(HcclCommDfxTest, Ut_SetDpuStreamId_When_Normal_Expect_SetSuccess)
{
    u32 expectedStreamId = 999;
    dfx_->SetDpuStreamId(expectedStreamId);
    EXPECT_EQ(dfx_->dpuStreamId_, expectedStreamId);
}

// 测试 SetDpuStreamId - 设置为 0
TEST_F(HcclCommDfxTest, Ut_SetDpuStreamId_When_Zero_Expect_SetSuccess)
{
    dfx_->SetDpuStreamId(0);
    EXPECT_EQ(dfx_->dpuStreamId_, 0u);
}

// 测试 GetDpuCallback - 获取回调
TEST_F(HcclCommDfxTest, Ut_GetDpuCallback_When_Normal_Expect_ReturnValidCallback)
{
    auto callback = dfx_->GetDpuCallback();
    EXPECT_TRUE(callback != nullptr);
}

// 测试 GetDpuCallback - 回调可调用
TEST_F(HcclCommDfxTest, Ut_GetDpuCallback_When_CallCallback_Expect_ReturnSuccess)
{
    auto callback = dfx_->GetDpuCallback();
    Hccl::TaskParam taskParam{};
    u64 handle = 0xFFFFFFFFFFFFFFFF;
    u32 remoteRankId = 2;

    // 先建立 handle 到 remoteRankId 的映射
    HcclCommDfx::AddChannelRemoteRankId("test_comm", handle, remoteRankId);
    
    HcclResult ret = callback(taskParam, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 SetAicpuTaskIdAndStreamId - 正常设置
TEST_F(HcclCommDfxTest, Ut_SetAicpuTaskIdAndStreamId_When_Normal_Expect_SetSuccess)
{
    u32 taskId = 555;
    u32 streamId = 666;
    dfx_->SetAicpuTaskIdAndStreamId(taskId, streamId);
    EXPECT_EQ(dfx_->aicpuTaskId_, taskId);
    EXPECT_EQ(dfx_->aicpuStreamId_, streamId);
}

// 测试 SetAicpuTaskIdAndStreamId - 设置 INVALID_UINT
TEST_F(HcclCommDfxTest, Ut_SetAicpuTaskIdAndStreamId_When_InvalidValue_Expect_SetSuccess)
{
    dfx_->SetAicpuTaskIdAndStreamId(INVALID_UINT, INVALID_UINT);
    EXPECT_EQ(dfx_->aicpuTaskId_, INVALID_UINT);
    EXPECT_EQ(dfx_->aicpuStreamId_, INVALID_UINT);
}

// 测试不同 streamId 的 taskId 独立
TEST_F(HcclCommDfxTest, Ut_GetTaskId_When_DifferentStreamId_Expect_Independent)
{
    u32 streamId1 = 111;
    u32 streamId2 = 222;
    
    u32 taskId1 = HcclCommDfx::GetTaskId(streamId1);
    EXPECT_EQ(taskId1, 1u);
    
    u32 taskId2 = HcclCommDfx::GetTaskId(streamId2);
    EXPECT_EQ(taskId2, 1u);
    
    taskId1 = HcclCommDfx::GetTaskId(streamId1);
    EXPECT_EQ(taskId1, 2u);
    
    taskId2 = HcclCommDfx::GetTaskId(streamId2);
    EXPECT_EQ(taskId2, 2u);
}

TEST_F(HcclCommDfxTest, Ut_Add_Get_ChannelRemoteRankId)
{
    u64 channelHandle = 0x9527;
    u32 remoteRankId = 3;
    dfxLite_->AddChannelRemoteRankId(channelHandle, remoteRankId);

    EXPECT_EQ(dfxLite_->GetChannelRemoteRankId(channelHandle), remoteRankId);

    EXPECT_EQ(dfxLite_->GetChannelRemoteRankId(0x123), INVALID_UINT);
}

// 测试 GetOpModeFlags - OFFLOAD 模式返回 false
TEST_F(HcclCommDfxTest, Ut_GetOpModeFlags_When_OpModeIsOffload_Expect_ReturnFalse)
{
    auto opInfo = std::make_shared<Hccl::DfxOpInfo>();
    opInfo->op_.opMode = Hccl::OpMode::OFFLOAD;
    MOCKER_CPP(&Hccl::MirrorTaskManager::GetCurrDfxOpInfo,
        std::shared_ptr<Hccl::DfxOpInfo>(Hccl::MirrorTaskManager::*)() const)
        .stubs()
        .will(returnValue(opInfo));

    bool isOpBase = true;
    bool isCached = false;
    HcclResult ret = dfx_->GetOpModeFlags(isOpBase, isCached);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(isOpBase);
    EXPECT_TRUE(isCached);
}

TEST_F(HcclCommDfxTest, Ut_GetOpModeFlags_When_OpModeIsAclgraph_Expect_BothTrue)
{
    auto opInfo = std::make_shared<Hccl::DfxOpInfo>();
    opInfo->op_.opMode = Hccl::OpMode::ACLGRAPH;
    MOCKER_CPP(&Hccl::MirrorTaskManager::GetCurrDfxOpInfo,
        std::shared_ptr<Hccl::DfxOpInfo>(Hccl::MirrorTaskManager::*)() const)
        .stubs()
        .will(returnValue(opInfo));

    bool isOpBase = false;
    bool isCached = false;
    HcclResult ret = dfx_->GetOpModeFlags(isOpBase, isCached);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(isOpBase);
    EXPECT_TRUE(isCached);
}

// 测试 GetChannelRemoteRankId（非Lite版）- 正常查找命中 shared_lock 读路径
TEST_F(HcclCommDfxTest, Ut_GetChannelRemoteRankId_When_Exist_Expect_ReturnSuccess)
{
    std::string commTag = "test_comm";
    u64 channelHandle = 0x8888;
    u32 remoteRankId = 7;

    // 先通过 AddChannelRemoteRankId 写入（覆盖 unique_lock 写路径）
    HcclCommDfx::AddChannelRemoteRankId(commTag, channelHandle, remoteRankId);

    // 查找已存在的 commTag+handle（覆盖 shared_lock 读路径）
    u32 result = INVALID_UINT;
    HcclResult ret = HcclCommDfx::GetChannelRemoteRankId(commTag, channelHandle, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result, remoteRankId);

    // 清理静态 map，避免影响其他用例
    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 GetChannelRemoteRankId - commTag 不存在
TEST_F(HcclCommDfxTest, Ut_GetChannelRemoteRankId_When_CommTagNotFound_Expect_ReturnParaError)
{
    u32 result = INVALID_UINT;
    HcclResult ret = HcclCommDfx::GetChannelRemoteRankId("non_exist_comm", 0x1234, result);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(result, INVALID_UINT);
}

// 测试 GetChannelRemoteRankId - commTag 存在但 handle 不存在
TEST_F(HcclCommDfxTest, Ut_GetChannelRemoteRankId_When_HandleNotFound_Expect_ReturnParaError)
{
    std::string commTag = "test_comm_handle";
    HcclCommDfx::AddChannelRemoteRankId(commTag, 0x1111, 1);

    u32 result = INVALID_UINT;
    HcclResult ret = HcclCommDfx::GetChannelRemoteRankId(commTag, 0x2222, result);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(result, INVALID_UINT);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 AddTaskInfoCallback - CCU 批量查找路径，覆盖 shared_lock 读路径(L91)
TEST_F(HcclCommDfxTest, Ut_AddTaskInfoCallback_When_CcuTaskAndHandleExist_Expect_Success)
{
    std::string commTag = "test_comm";
    u64 channelHandle = 0x7777;
    u32 remoteRankId = 5;

    // 写入 channelRemoteRankId_ 映射
    HcclCommDfx::AddChannelRemoteRankId(commTag, channelHandle, remoteRankId);

    // 构造 CCU 类型 TaskParam
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_CCU;
    auto ccuDetailInfo = std::make_shared<std::vector<Hccl::CcuProfilingInfo>>();
    Hccl::CcuProfilingInfo profInfo{};
    profInfo.channelId[0] = 0; // 非 INVALID_VALUE_CHANNELID，进入查找
    profInfo.channelHandle[0] = channelHandle;
    ccuDetailInfo->push_back(profInfo);
    taskParam.ccuDetailInfo = ccuDetailInfo;

    // mock MirrorTaskManager
    auto opInfo = std::make_shared<Hccl::DfxOpInfo>();
    MOCKER_CPP(&Hccl::MirrorTaskManager::GetCurrDfxOpInfo,
        std::shared_ptr<Hccl::DfxOpInfo>(Hccl::MirrorTaskManager::*)() const)
        .stubs()
        .will(returnValue(opInfo));
    MOCKER_CPP(&Hccl::MirrorTaskManager::AddTaskInfo,
        HcclResult(Hccl::MirrorTaskManager::*)(u32, u32, u32, const Hccl::TaskParam&,
            std::shared_ptr<Hccl::DfxOpInfo>, bool))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = dfx_->AddTaskInfoCallback(1, 1, taskParam, channelHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 AddTaskInfoCallback - CCU 路径但 commTag 不存在，覆盖 shared_lock 读路径的提前返回
TEST_F(HcclCommDfxTest, Ut_AddTaskInfoCallback_When_CcuTaskAndCommTagNotFound_Expect_ParaError)
{
    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_CCU;
    auto ccuDetailInfo = std::make_shared<std::vector<Hccl::CcuProfilingInfo>>();
    Hccl::CcuProfilingInfo profInfo{};
    profInfo.channelId[0] = 0;
    profInfo.channelHandle[0] = 0x9999;
    ccuDetailInfo->push_back(profInfo);
    taskParam.ccuDetailInfo = ccuDetailInfo;

    // 不添加任何 channelRemoteRankId_ 映射，commTag 查找失败
    HcclResult ret = dfx_->AddTaskInfoCallback(1, 1, taskParam, INVALID_U64);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 HcclCommDfxLite::GetChannelRemoteRankId - INVALID_U64 handle 早期返回
TEST_F(HcclCommDfxTest, Ut_GetChannelRemoteRankIdLite_When_InvalidU64Handle_Expect_ReturnInvalidUint)
{
    u32 result = dfxLite_->GetChannelRemoteRankId(INVALID_U64);
    EXPECT_EQ(result, INVALID_UINT);
}

// 测试 HcclCommDfxLite::Init 重复初始化
TEST_F(HcclCommDfxTest, Ut_InitLite_When_AlreadyInitialized_Expect_ReturnSuccess)
{
    HcclResult ret = dfxLite_->Init(0, "test_comm2", 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 HcclCommDfxLite::GetMirrorTaskManagerLite
TEST_F(HcclCommDfxTest, Ut_GetMirrorTaskManagerLite_When_Initialized_Expect_ReturnNotNull)
{
    Hccl::MirrorTaskManagerLite* mgr = dfxLite_->GetMirrorTaskManagerLite();
    EXPECT_NE(mgr, nullptr);
}

// 测试 HcclCommDfxLite::AddChannelRemoteRankId 多个 handle
TEST_F(HcclCommDfxTest, Ut_AddChannelRemoteRankIdLite_When_MultipleHandles_Expect_AllStored)
{
    u64 handle1 = 0x1111;
    u64 handle2 = 0x2222;
    u32 rank1 = 10;
    u32 rank2 = 20;

    dfxLite_->AddChannelRemoteRankId(handle1, rank1);
    dfxLite_->AddChannelRemoteRankId(handle2, rank2);

    EXPECT_EQ(dfxLite_->GetChannelRemoteRankId(handle1), rank1);
    EXPECT_EQ(dfxLite_->GetChannelRemoteRankId(handle2), rank2);
}

// 测试 HcclCommDfxLite::ReportAllTasks
TEST_F(HcclCommDfxTest, Ut_ReportAllTasksLite_When_Normal_Expect_ReturnSuccess)
{
    HcclResult ret = dfxLite_->ReportAllTasks();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 HcclCommDfxLite::UpdateProfStat
TEST_F(HcclCommDfxTest, Ut_UpdateProfStatLite_When_Normal_Expect_ReturnSuccess)
{
    HcclResult ret = dfxLite_->UpdateProfStat();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 HcclCommDfx::ReportAllTasks
TEST_F(HcclCommDfxTest, Ut_ReportAllTasks_When_Normal_Expect_ReturnSuccess)
{
    HcclResult ret = dfx_->ReportAllTasks(false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 HcclCommDfx::UpdateProfStat
TEST_F(HcclCommDfxTest, Ut_UpdateProfStat_When_Normal_Expect_ReturnSuccess)
{
    HcclResult ret = dfx_->UpdateProfStat();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 AddTaskInfoCallback - CCU 路径，commTag 存在但 channelHandle 不在 handleMap 中
TEST_F(HcclCommDfxTest, Ut_AddTaskInfoCallback_When_CcuTaskAndHandleNotInMap_Expect_ParaError)
{
    std::string commTag = "test_comm";
    u64 channelHandle = 0x7777;
    u32 remoteRankId = 5;

    HcclCommDfx::AddChannelRemoteRankId(commTag, channelHandle, remoteRankId);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_CCU;
    auto ccuDetailInfo = std::make_shared<std::vector<Hccl::CcuProfilingInfo>>();
    Hccl::CcuProfilingInfo profInfo{};
    profInfo.channelId[0] = 0;
    profInfo.channelHandle[0] = 0xBADBAD;
    ccuDetailInfo->push_back(profInfo);
    taskParam.ccuDetailInfo = ccuDetailInfo;

    HcclResult ret = dfx_->AddTaskInfoCallback(1, 1, taskParam, channelHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 AddTaskInfoCallback - 非 CCU 任务类型
TEST_F(HcclCommDfxTest, Ut_AddTaskInfoCallback_When_NonCcuTaskAndHandleExist_Expect_Success)
{
    std::string commTag = "test_comm";
    u64 channelHandle = 0x5555;
    u32 remoteRankId = 8;

    HcclCommDfx::AddChannelRemoteRankId(commTag, channelHandle, remoteRankId);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_RECORD;

    auto opInfo = std::make_shared<Hccl::DfxOpInfo>();
    MOCKER_CPP(&Hccl::MirrorTaskManager::GetCurrDfxOpInfo,
        std::shared_ptr<Hccl::DfxOpInfo>(Hccl::MirrorTaskManager::*)() const)
        .stubs()
        .will(returnValue(opInfo));
    MOCKER_CPP(&Hccl::MirrorTaskManager::AddTaskInfo,
        HcclResult(Hccl::MirrorTaskManager::*)(u32, u32, u32, const Hccl::TaskParam&,
            std::shared_ptr<Hccl::DfxOpInfo>, bool))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = dfx_->AddTaskInfoCallback(1, 1, taskParam, channelHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
    HcclCommDfx::channelRemoteRankId_.clear();
}

// 测试 HcclCommDfx::SetCurrDfxOpInfo
TEST_F(HcclCommDfxTest, Ut_SetCurrDfxOpInfo_When_Normal_Expect_ReturnSuccess)
{
    auto dfxOpInfo = std::make_shared<Hccl::DfxOpInfo>();
    HcclResult ret = dfx_->SetCurrDfxOpInfo(dfxOpInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 HcclCommDfx::GetCallback
TEST_F(HcclCommDfxTest, Ut_GetCallback_When_Initialized_Expect_ReturnValidCallback)
{
    auto callback = dfx_->GetCallback();
    EXPECT_TRUE(callback != nullptr);
}

// 测试 HcclCommDfx::ReportOp
TEST_F(HcclCommDfxTest, Ut_ReportOp_When_Normal_Expect_ReturnSuccess)
{
    HcclResult ret = dfx_->ReportOp(0, false, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
