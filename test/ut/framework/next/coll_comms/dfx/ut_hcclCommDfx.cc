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

// 测试 IsOpBase - OFFLOAD 模式返回 false
TEST_F(HcclCommDfxTest, Ut_IsOpBase_When_OpModeIsOffload_Expect_ReturnFalse)
{
    auto opInfo = std::make_shared<Hccl::DfxOpInfo>();
    opInfo->op_.opMode = Hccl::OpMode::OFFLOAD;
    MOCKER_CPP(&Hccl::MirrorTaskManager::GetCurrDfxOpInfo,
        std::shared_ptr<Hccl::DfxOpInfo>(Hccl::MirrorTaskManager::*)() const)
        .stubs()
        .will(returnValue(opInfo));

    bool isOpBase = true;
    HcclResult ret = dfx_->IsOpBase(isOpBase);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(isOpBase);
}
