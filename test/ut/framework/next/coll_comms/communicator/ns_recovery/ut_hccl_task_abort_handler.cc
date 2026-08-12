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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#include "coll_comm.h"
#include "ns_recovery/task_abort_handler.h"
#include "ccu_device_pub.h"
#undef private

using namespace hccl;
using namespace hcomm;

class HcclTaskAbortHandlerTest : public testing::Test {
public:
    HcclTaskAbortHandler handler;

    static void SetUpTestCase() { std::cout << "HcclTaskAbortHandlerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclTaskAbortHandlerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        handler.commVector_.clear();
        std::cout << "A Test case in HcclTaskAbortHandlerTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in HcclTaskAbortHandlerTest TearDown" << std::endl;
        handler.commVector_.clear();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_success)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm* comm = nullptr;
    handler.Register(comm);
    void* args = reinterpret_cast<void*>(&handler.commVector_);

    // 模拟 Suspend 方法返回成功
    MOCKER_CPP(&CollComm::Suspend, HcclResult(CollComm::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 清理
    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_with_null_args)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 30U;

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, nullptr);
    EXPECT_NE(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));
}

TEST_F(HcclTaskAbortHandlerTest, test_unregister_not_found_comm)
{
    CollComm* comm1 = reinterpret_cast<CollComm*>(0x1000);
    CollComm* comm2 = reinterpret_cast<CollComm*>(0x2000);

    handler.Register(comm1);
    HcclResult ret = handler.UnRegister(comm2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    handler.UnRegister(comm1);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_suspending)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 0U;

    CollComm* comm = nullptr;
    handler.Register(comm);
    void* args = reinterpret_cast<void*>(&handler.commVector_);

    MOCKER_CPP(&CollComm::Suspend, HcclResult(CollComm::*)()).stubs().will(returnValue(HCCL_E_SUSPENDING));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_suspending)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 0U;

    CollComm* comm = nullptr;
    handler.Register(comm);
    void* args = reinterpret_cast<void*>(&handler.commVector_);

    MOCKER(CcuIsInited).stubs().will(returnValue(true));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm::*)()).stubs().will(returnValue(HCCL_E_SUSPENDING));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_post_ccu_set_task_kill_fail)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 0U;

    CollComm* comm = nullptr;
    handler.Register(comm);
    void* args = reinterpret_cast<void*>(&handler.commVector_);

    MOCKER(CcuIsInited).stubs().will(returnValue(true));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_E_INTERNAL));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_NE(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_register_and_unregister_multiple)
{
    CollComm* comm1 = reinterpret_cast<CollComm*>(0x1000);
    CollComm* comm2 = reinterpret_cast<CollComm*>(0x2000);

    HcclResult ret = handler.Register(comm1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = handler.Register(comm2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(handler.commVector_.size(), 2u);

    ret = handler.UnRegister(comm1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler.commVector_.size(), 1u);

    ret = handler.UnRegister(comm2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler.commVector_.size(), 0u);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_fail)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm* comm = nullptr;
    handler.Register(comm);
    void* args = reinterpret_cast<void*>(&handler.commVector_);

    // 模拟 Suspend 方法返回失败
    MOCKER_CPP(&CollComm::Suspend, HcclResult(CollComm::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 清理
    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_success)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm* comm = nullptr;
    handler.Register(comm);
    void* args = reinterpret_cast<void*>(&handler.commVector_);

    // 模拟 CCU 已初始化及相关函数返回成功
    MOCKER(CcuIsInited).stubs().will(returnValue(true));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));

    // 模拟 Clean 方法返回成功
    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 清理
    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_post_skip_taskkill_when_ccu_not_inited)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm* comm = nullptr;
    handler.Register(comm);
    void* args = reinterpret_cast<void*>(&handler.commVector_);

    // 模拟 CCU 未初始化，应跳过 TaskKill
    MOCKER(CcuIsInited).stubs().will(returnValue(false));

    // 模拟 Clean 方法返回成功
    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 清理
    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_fail)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm* comm = nullptr;
    handler.Register(comm);
    void* args = reinterpret_cast<void*>(&handler.commVector_);

    // 模拟 CCU 已初始化及相关函数返回成功
    MOCKER(CcuIsInited).stubs().will(returnValue(true));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));

    // 模拟 Clean 方法返回失败
    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 清理
    handler.UnRegister(comm);
}
