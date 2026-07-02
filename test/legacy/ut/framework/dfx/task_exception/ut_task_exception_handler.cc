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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>

#include "const_val.h"
#include "orion_adapter_rts.h"
#include "ccu_error_info.h"
#include "global_mirror_tasks.h"
#include "mirror_task_manager.h"
#include "ccu_dfx.h"
#include "mc2_global_mirror_tasks.h"
#include "ccu_device_manager.h"
#include "ccu_component.h"

#define private public
#define protected public
#include "task_exception_handler.h"
#include "communicator_impl.h"
#include "coll_service_device_mode.h"
#include "ccu_transport_manager.h"
#include "mc2_compont.h"
#undef private
#undef protected


using namespace std;
using namespace Hccl;
using namespace CcuRep;

class TaskExceptionHandlerTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TaskExceptionHandlerTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TaskExceptionHandlerTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in TaskExceptionHandlerTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in TaskExceptionHandlerTest TearDown" << std::endl;
    }

    unique_ptr<TaskInfo> InitTaskInfo(u32 streamId = 0, u32 taskId = 0, u32 remoteRank = 0)
    {
        TaskParam taskParam{};
        shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
        return make_unique<TaskInfo>(streamId, taskId, remoteRank, taskParam, dfxOpInfo);
    }
};

TEST_F(TaskExceptionHandlerTest, TestGetInstanceWithInvalidDevId)
{
    const uint32_t HANDEL_EXCEPTION_NUM = 128; // 大于最大设备号
    size_t invalidDevId = HANDEL_EXCEPTION_NUM + 1;
    EXPECT_EQ(nullptr, TaskExceptionHandlerManager::GetHandler(invalidDevId));
}

TEST_F(TaskExceptionHandlerTest, TestGetInstanceWithValidDevId)
{
    size_t validDevId = 0;
    EXPECT_NE(nullptr, TaskExceptionHandlerManager::GetHandler(validDevId));
}

TEST_F(TaskExceptionHandlerTest, TestConstructorAndDestructor)
{
    TaskExceptionHandler *instance = new TaskExceptionHandler(0);
    EXPECT_NO_THROW(instance->~TaskExceptionHandler());
    delete instance;
}

TEST_F(TaskExceptionHandlerTest, TestRegisterAndUnRegister)
{
    TaskExceptionHandler *instance = TaskExceptionHandlerManager::GetHandler(0);
    instance->Register();
    instance->UnRegister();
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_default)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::DEFAULT;
    ccuErrorInfo.repType = CcuRepType::JUMP;
    ccuErrorInfo.instrId = 0xffff;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: CcuErrorType[CcuErrorType::DEFAULT]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_mission)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::MISSION;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 10;
    const string  statusMsg = "Transaction ACK Timeout";
    strncpy_s(ccuErrorInfo.msg.mission.missionError, MISSION_STATUS_MSG_LEN, statusMsg.c_str(), statusMsg.length());

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[10]: dieId[0], missionId[1], missionError[Transaction ACK Timeout]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_loop)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::LOOP;
    ccuErrorInfo.repType = CcuRepType::LOOP;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.loop.startInstrId = 7;
    ccuErrorInfo.msg.loop.endInstrId = 17;
    ccuErrorInfo.msg.loop.loopCnt = 10;
    ccuErrorInfo.msg.loop.loopCurrentCnt = 8;
    ccuErrorInfo.msg.loop.addrStride = 0xaabbcc;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Loop startInstrId[7], endInstrId[17], executorId[0], totalIter[10], curIter[8], addressStride[0xaabbcc]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_loop_group)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::LOOP_GROUP;
    ccuErrorInfo.repType = CcuRepType::LOOPGROUP;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.loopGroup.startLoopInsId = 17;
    ccuErrorInfo.msg.loopGroup.loopInsCnt = 5;
    ccuErrorInfo.msg.loopGroup.expandOffset = 3;
    ccuErrorInfo.msg.loopGroup.expandCnt = 2;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: LoopGroup startLoopInsId[17], loopInsCnt[5], expandOffset[3], expandCnt[2]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_loc_post_sem)
{
    auto taskInfo = InitTaskInfo();
    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::LOC_POST_SEM;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalValue = 0xabc;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Set sem[11], semValue[0x0abc], mask[0x0010]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_loc_wait_sem)
{
    auto taskInfo = InitTaskInfo();
    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::LOC_WAIT_SEM;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalValue = 0xabc;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Wait sem[11], semValue[0x0abc], mask[0x0010]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_rem_post_sem)
{
    auto taskInfo = InitTaskInfo();
    MOCKER(TaskExceptionHandler::GetRankIdByChannelId).stubs().with(eq(static_cast<uint16_t>(1)), mockcpp::any()).will(returnValue(100));

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::REM_POST_SEM;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;
    ccuErrorInfo.msg.waitSignal.channelId[0] = 1;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Post, Use sem[11], mask[0x0010], rankId[100]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_rem_wait_sem)
{
    auto taskInfo = InitTaskInfo();
    MOCKER(TaskExceptionHandler::GetRankIdByChannelId).stubs().with(eq(static_cast<uint16_t>(1)), mockcpp::any()).will(returnValue(100));

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::REM_WAIT_SEM;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalValue = 0xabc;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;
    ccuErrorInfo.msg.waitSignal.channelId[0] = 1;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Wait, Use sem[11], semValue[0x0abc], mask[0x0010], rankId[100]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_rem_post_var)
{
    auto taskInfo = InitTaskInfo();
    MOCKER(TaskExceptionHandler::GetRankIdByChannelId).stubs().with(eq(static_cast<uint16_t>(1)), mockcpp::any()).will(returnValue(100));

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::REM_POST_VAR;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.waitSignal.paramValue = 0xaaaabbbbcccc;
    ccuErrorInfo.msg.waitSignal.paramId = 0xa;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;
    ccuErrorInfo.msg.waitSignal.channelId[0] = 1;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Post Variable[0x0000aaaabbbbcccc] To Param[10], Use sem[11], mask[0x0010], rankId[100]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_rem_wait_group)
{
    auto taskInfo = InitTaskInfo();
    MOCKER(TaskExceptionHandler::GetRankIdByChannelId).stubs().with(mockcpp::any(), mockcpp::any())
        .will(returnValue(100)).then(returnValue(200)).then(returnValue(300)).then(returnValue(400));

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::REM_WAIT_GROUP;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalValue = 0xabc;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;
    ccuErrorInfo.msg.waitSignal.channelId[0] = 1;
    ccuErrorInfo.msg.waitSignal.channelId[1] = 2;
    ccuErrorInfo.msg.waitSignal.channelId[2] = 3;
    ccuErrorInfo.msg.waitSignal.channelId[3] = 4;
    for (uint32_t i = 4; i < WAIT_SIGNAL_CHANNEL_SIZE; ++i) {
        ccuErrorInfo.msg.waitSignal.channelId[i] = 0xffff;
    }

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Wait Group, Use sem[11], semValue[0x0abc], mask[0x0010], rankIds[100, 200, 300, 400]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_post_shared_var)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::POST_SHARED_VAR;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.waitSignal.paramId = 0xa;
    ccuErrorInfo.msg.waitSignal.paramValue = 0xaaaabbbbcccc;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Post Shared Variable[10] from Variable[0x0000aaaabbbbcccc], Use sem[11], mask[0x0010]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_post_shared_sem)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::POST_SHARED_SEM;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Post, Use sem[11], mask[0x0010]");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_read)
{
    auto taskInfo = InitTaskInfo();
    MOCKER(TaskExceptionHandler::GetRankIdByChannelId).stubs().with(eq(static_cast<uint16_t>(1)), mockcpp::any()).will(returnValue(100));

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::TRANS_MEM;
    ccuErrorInfo.repType = CcuRepType::READ;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.transMem.rmtAddr = 0xaaaa;
    ccuErrorInfo.msg.transMem.rmtToken = 0xbbbb;
    ccuErrorInfo.msg.transMem.locAddr = 0xcccc;
    ccuErrorInfo.msg.transMem.locToken = 0xdddd;
    ccuErrorInfo.msg.transMem.signalId = 0xb;
    ccuErrorInfo.msg.transMem.signalMask = 0x0010;
    ccuErrorInfo.msg.transMem.channelId = 1;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Read Memory[0x000000000000aaaa] To Memory[0x000000000000cccc], Len[0], Set sem[11] with mask[0x0010], remoteRankId[100], srcEID[IpAddress[0000:0000:0000:0000:0000:0000:0000:0000, AF=IPv4, addr=0.0.0.0]], dstEID[IpAddress[0000:0000:0000:0000:0000:0000:0000:0000, AF=IPv4, addr=0.0.0.0]] ccu transMem Len[0]B > 256MB or is zero, not support!");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_write)
{
    auto taskInfo = InitTaskInfo();
    MOCKER(TaskExceptionHandler::GetRankIdByChannelId).stubs().with(eq(static_cast<uint16_t>(1)), mockcpp::any()).will(returnValue(100));

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::TRANS_MEM;
    ccuErrorInfo.repType = CcuRepType::WRITE;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.transMem.rmtAddr = 0xaaaa;
    ccuErrorInfo.msg.transMem.rmtToken = 0xbbbb;
    ccuErrorInfo.msg.transMem.locAddr = 0xcccc;
    ccuErrorInfo.msg.transMem.locToken = 0xdddd;
    ccuErrorInfo.msg.transMem.signalId = 0xb;
    ccuErrorInfo.msg.transMem.signalMask = 0x0010;
    ccuErrorInfo.msg.transMem.channelId = 1;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Write Memory[0x000000000000cccc] to Memory[0x000000000000aaaa], Len[0], Set sem[11] with mask[0x0010], remoteRankId[100], srcEID[IpAddress[0000:0000:0000:0000:0000:0000:0000:0000, AF=IPv4, addr=0.0.0.0]], dstEID[IpAddress[0000:0000:0000:0000:0000:0000:0000:0000, AF=IPv4, addr=0.0.0.0]] ccu transMem Len[0]B > 256MB or is zero, not support!");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_local_cpy)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::TRANS_MEM;
    ccuErrorInfo.repType = CcuRepType::LOCAL_CPY;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.transMem.rmtAddr = 0xaaaa;
    ccuErrorInfo.msg.transMem.rmtToken = 0xbbbb;
    ccuErrorInfo.msg.transMem.locAddr = 0xcccc;
    ccuErrorInfo.msg.transMem.locToken = 0xdddd;
    ccuErrorInfo.msg.transMem.signalId = 0xb;
    ccuErrorInfo.msg.transMem.signalMask = 0x0010;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Read Memory[0x000000000000cccc] to Memory[0x000000000000aaaa], Len[0], Set sem[11] with mask[0x0010] ccu transMem Len[0]B > 256MB or is zero, not support!");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_local_reduce)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::TRANS_MEM;
    ccuErrorInfo.repType = CcuRepType::LOCAL_REDUCE;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.transMem.rmtAddr = 0xaaaa;
    ccuErrorInfo.msg.transMem.rmtToken = 0xbbbb;
    ccuErrorInfo.msg.transMem.locAddr = 0xcccc;
    ccuErrorInfo.msg.transMem.locToken = 0xdddd;
    ccuErrorInfo.msg.transMem.signalId = 0xb;
    ccuErrorInfo.msg.transMem.signalMask = 0x0010;
    ccuErrorInfo.msg.transMem.dataType = 1;
    ccuErrorInfo.msg.transMem.opType = 2;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Read Memory[0x000000000000cccc] to Memory[0x000000000000aaaa], Len[0], Set sem[11] with mask[0x0010], dataType[1], opType[2] ccu transMem Len[0]B > 256MB or is zero, not support!");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_buf_read)
{
    auto taskInfo = InitTaskInfo();
    MOCKER(TaskExceptionHandler::GetRankIdByChannelId).stubs().with(eq(static_cast<uint16_t>(1)), mockcpp::any()).will(returnValue(100));

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::BUF_TRANS_MEM;
    ccuErrorInfo.repType = CcuRepType::BUF_READ;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.bufTransMem.addr = 0xaaaa;
    ccuErrorInfo.msg.bufTransMem.token = 0xbbbb;
    ccuErrorInfo.msg.bufTransMem.bufId = 0xa;
    ccuErrorInfo.msg.bufTransMem.signalId = 0xb;
    ccuErrorInfo.msg.bufTransMem.signalMask = 0x0010;
    ccuErrorInfo.msg.bufTransMem.channelId = 1;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Read Rmt Mem[0x000000000000aaaa] To CcuBuffer[10], Len[0], sem[11], mask[0x0010], remoteRankId[100], srcEID[IpAddress[0000:0000:0000:0000:0000:0000:0000:0000, AF=IPv4, addr=0.0.0.0]], dstEID[IpAddress[0000:0000:0000:0000:0000:0000:0000:0000, AF=IPv4, addr=0.0.0.0]] ccu transMem Len[0]B > 256MB or is zero, not support!");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_buf_write)
{
    auto taskInfo = InitTaskInfo();
    MOCKER(TaskExceptionHandler::GetRankIdByChannelId).stubs().with(eq(static_cast<uint16_t>(1)), mockcpp::any()).will(returnValue(100));

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::BUF_TRANS_MEM;
    ccuErrorInfo.repType = CcuRepType::BUF_WRITE;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.bufTransMem.addr = 0xaaaa;
    ccuErrorInfo.msg.bufTransMem.token = 0xbbbb;
    ccuErrorInfo.msg.bufTransMem.bufId = 0xa;
    ccuErrorInfo.msg.bufTransMem.signalId = 0xb;
    ccuErrorInfo.msg.bufTransMem.signalMask = 0x0010;
    ccuErrorInfo.msg.bufTransMem.channelId = 1;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Write CcuBuffer[10] To Rmt Mem[0x000000000000aaaa], Len[0], sem[11], mask[0x0010], remoteRankId[100], srcEID[IpAddress[0000:0000:0000:0000:0000:0000:0000:0000, AF=IPv4, addr=0.0.0.0]], dstEID[IpAddress[0000:0000:0000:0000:0000:0000:0000:0000, AF=IPv4, addr=0.0.0.0]] ccu transMem Len[0]B > 256MB or is zero, not support!");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_buf_loc_read)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::BUF_TRANS_MEM;
    ccuErrorInfo.repType = CcuRepType::BUF_LOC_READ;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.bufTransMem.addr = 0xaaaa;
    ccuErrorInfo.msg.bufTransMem.token = 0xbbbb;
    ccuErrorInfo.msg.bufTransMem.bufId = 0xa;
    ccuErrorInfo.msg.bufTransMem.signalId = 0xb;
    ccuErrorInfo.msg.bufTransMem.signalMask = 0x0010;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Read Loc Mem[0x000000000000aaaa] To CcuBuffer[10], Len[0], sem[11], mask[0x0010] ccu transMem Len[0]B > 256MB or is zero, not support!");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_buf_loc_write)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::BUF_TRANS_MEM;
    ccuErrorInfo.repType = CcuRepType::BUF_LOC_WRITE;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.bufTransMem.addr = 0xaaaa;
    ccuErrorInfo.msg.bufTransMem.token = 0xbbbb;
    ccuErrorInfo.msg.bufTransMem.bufId = 0xa;
    ccuErrorInfo.msg.bufTransMem.signalId = 0xb;
    ccuErrorInfo.msg.bufTransMem.signalMask = 0x0010;

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Write CcuBuffer[10] To Loc Mem[0x000000000000aaaa], Len[0], sem[11], mask[0x0010] ccu transMem Len[0]B > 256MB or is zero, not support!");
}

TEST_F(TaskExceptionHandlerTest, test_ccu_error_msg_when_type_is_buf_reduce)
{
    auto taskInfo = InitTaskInfo();

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::BUF_REDUCE;
    ccuErrorInfo.repType = CcuRepType::BUF_REDUCE;
    ccuErrorInfo.instrId = 0xffff;
    ccuErrorInfo.msg.bufReduce.count = 4;
    ccuErrorInfo.msg.bufReduce.dataType = 1;
    ccuErrorInfo.msg.bufReduce.outputDataType = 2;
    ccuErrorInfo.msg.bufReduce.opType = 3;
    ccuErrorInfo.msg.bufReduce.signalId = 0xb;
    ccuErrorInfo.msg.bufReduce.signalMask = 0x0010;
    ccuErrorInfo.msg.bufReduce.bufIds[0] = 100;
    ccuErrorInfo.msg.bufReduce.bufIds[1] = 200;
    ccuErrorInfo.msg.bufReduce.bufIds[2] = 300;
    ccuErrorInfo.msg.bufReduce.bufIds[3] = 400;
    for (uint32_t i = 4; i < BUF_REDUCE_ID_SIZE; ++i) {
        ccuErrorInfo.msg.bufReduce.bufIds[i] = 0xffff;
    }

    auto msg = TaskExceptionHandler::GetCcuErrorMsgByType(ccuErrorInfo, *taskInfo);
    EXPECT_EQ(msg, "InstrId[65535]: Buffer Reduce count[4], dataType[1], outputDataType[2], opType[3], sem[11], mask[0x0010], CcuBuffers[100, 200, 300, 400]");
}

TEST_F(TaskExceptionHandlerTest, test_get_rank_id_by_channel_id)
{
    auto taskInfo = InitTaskInfo();

    taskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    EXPECT_EQ(TaskExceptionHandler::GetRankIdByChannelId(1, *taskInfo), INVALID_RANKID);    // task type error

    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    EXPECT_EQ(TaskExceptionHandler::GetRankIdByChannelId(1, *taskInfo), INVALID_RANKID);    // communicator is nullptr

    taskInfo->dfxOpInfo_ = make_shared<DfxOpInfo>();
    // Mock CommunicatorImpl
    CommunicatorImpl communicator{};
    taskInfo->dfxOpInfo_->comm_ = &communicator;
    communicator.collServices[AcceleratorState::CCU_SCHED] = nullptr;
    EXPECT_EQ(TaskExceptionHandler::GetRankIdByChannelId(1, *taskInfo), INVALID_RANKID);    // Failed to get collService

    // Mock collService
    auto dieChannelId = make_pair(taskInfo->taskParam_.taskPara.Ccu.dieId, 1);
    communicator.collService = new CollServiceDeviceMode(&communicator);
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    communicator.collServices[AcceleratorState::CCU_SCHED] = std::make_shared<CollServiceDeviceMode>(&communicator);
    (static_cast<CollServiceDeviceMode*>(communicator.GetCcuCollService()))->
        GetCcuInsPreprocessor()->GetCcuComm()->GetCcuJettyMgr()->channelRemoteRankIdMap_.emplace(dieChannelId, 100);
    EXPECT_EQ(TaskExceptionHandler::GetRankIdByChannelId(1, *taskInfo), 100);
    delete communicator.collService;
}

TEST_F(TaskExceptionHandlerTest, ut_get_ccu_collservice_before_init_Expect_Throw_NullptrException)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo->dfxOpInfo_ = make_shared<DfxOpInfo>();
    CommunicatorImpl communicator{};
    taskInfo->dfxOpInfo_->comm_ = &communicator;
    EXPECT_THROW(TaskExceptionHandler::GetRankIdByChannelId(1, *taskInfo), NullPtrException);
}

TEST_F(TaskExceptionHandlerTest, test_get_group_rank_info)
{
    auto taskInfo = InitTaskInfo();

    EXPECT_EQ(TaskExceptionHandler::GetGroupRankInfo(*taskInfo), "");    // communicator is nullptr

    // Mock CommunicatorImpl
    CommunicatorImpl communicator{};
    communicator.id = "group_name";
    communicator.rankSize = 4;
    communicator.myRank = 1;
    taskInfo->dfxOpInfo_->comm_ = &communicator;
    EXPECT_EQ(TaskExceptionHandler::GetGroupRankInfo(*taskInfo), "group:[group_name], rankSize[4], rankId[1]");
}

TEST_F(TaskExceptionHandlerTest, test_process_when_task_more_than_50)
{
    // 打桩 GlobalMirrorTasks
    GlobalMirrorTasks &globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);  // diveceId 0
    shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
    dfxOpInfo->commIndex_ = 3;
    dfxOpInfo->op_.dataCount = 0xff;
    dfxOpInfo->op_.reduceOp = ReduceOp::PROD;
    dfxOpInfo->op_.dataType = DataType::FP64;
    dfxOpInfo->algType_ = AlgType{AlgType::RING}.Describe();
    dfxOpInfo->op_.inputMem = make_shared<Buffer>(0x111122223333, 0);
    dfxOpInfo->op_.outputMem = make_shared<Buffer>(0xaaaabbbbcccc, 0);
    CommunicatorImpl communicator{};    // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 4;
    communicator.myRank = 1;
    MOCKER(TaskExceptionHandler::PrintAicpuErrorMessage).stubs();
    dfxOpInfo->comm_ = &communicator;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    // 加入一些 Task 数据
    // 在异常 Task 前加入60个 Task
    for (uint32_t i = 0; i < 50; ++i) {
        auto preTaskInfo = InitTaskInfo(0, i); // streamId 0, taskId 0-59
        preTaskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
        preTaskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
        preTaskInfo->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
        mirrorTaskManager.AddTaskInfo(std::move(preTaskInfo));
    }
    // 加入当前异常 Task
    auto taskInfo = InitTaskInfo(0, 60); // streamId 0, taskId 60
    taskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    taskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskInfo->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));
    // 在异常 Task 后加入10个 Task
    for (uint32_t i = 0; i < 10; ++i) {
        auto postTaskInfo = InitTaskInfo(0, i + 61); // streamId 0, taskId 61-70
        postTaskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
        mirrorTaskManager.AddTaskInfo(std::move(postTaskInfo));
    }

    // 调用 TaskExceptionHandler::Process() 打印异常DFX信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 60;  // 当前异常TaskId 60
    TaskExceptionHandler::Process(&exceptionInfo);

    globalMirrorTasks.DestroyQueue(0, 0);   // diveceId 0, streamId 0
}

TEST_F(TaskExceptionHandlerTest, test_process_when_task_less_than_50)
{
    // 打桩 GlobalMirrorTasks
    GlobalMirrorTasks &globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);  // diveceId 0
    shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
    dfxOpInfo->commIndex_ = 3;
    dfxOpInfo->op_.dataCount = 0xff;
    dfxOpInfo->op_.reduceOp = ReduceOp::PROD;
    dfxOpInfo->op_.dataType = DataType::FP64;
    dfxOpInfo->algType_ =  AlgType{AlgType::RING}.Describe();
    dfxOpInfo->op_.inputMem = make_shared<Buffer>(0x111122223333, 0);
    dfxOpInfo->op_.outputMem = make_shared<Buffer>(0xaaaabbbbcccc, 0);
    CommunicatorImpl communicator{};    // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 4;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    // 加入一些 Task 数据
    // 在异常 Task 前加入一些 Task
    auto taskInfo0 = InitTaskInfo(0, 0);
    taskInfo0->taskParam_.taskType = TaskParamType::TASK_SDMA;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo0));
    // TASK_SDMA
    auto taskInfo1 = InitTaskInfo(0, 1);
    taskInfo1->taskParam_.taskType = TaskParamType::TASK_SDMA;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo1));
    // TASK_RDMA
    auto taskInfo2 = InitTaskInfo(0, 2);
    taskInfo2->taskParam_.taskType = TaskParamType::TASK_RDMA;
    taskInfo2->taskParam_.taskPara.DMA.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo2));
    // TASK_SEND_PAYLOAD
    auto taskInfo3 = InitTaskInfo(0, 3);
    taskInfo3->taskParam_.taskType = TaskParamType::TASK_SEND_PAYLOAD;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo3));
    // TASK_REDUCE_INLINE
    auto taskInfo4 = InitTaskInfo(0, 4);
    taskInfo4->taskParam_.taskType = TaskParamType::TASK_REDUCE_INLINE;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo4));
    // TASK_REDUCE_TBE
    auto taskInfo5 = InitTaskInfo(0, 5);
    taskInfo5->taskParam_.taskType = TaskParamType::TASK_REDUCE_TBE;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo5));
    // TASK_NOTIFY_RECORD
    auto taskInfo6 = InitTaskInfo(0, 6);
    taskInfo6->taskParam_.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskInfo6->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo6));
    // TASK_NOTIFY_WAIT
    auto taskInfo7 = InitTaskInfo(0, 7);
    taskInfo7->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskInfo7->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo7));
    // TASK_SEND_NOTIFY
    auto taskInfo8 = InitTaskInfo(0, 8);
    taskInfo8->taskParam_.taskType = TaskParamType::TASK_SEND_NOTIFY;
    taskInfo8->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo8));
    // TASK_WRITE_WITH_NOTIFY
    auto taskInfo9 = InitTaskInfo(0, 9);
    taskInfo9->taskParam_.taskType = TaskParamType::TASK_WRITE_WITH_NOTIFY;
    taskInfo9->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo9));
    // TASK_WRITE_REDUCE_WITH_NOTIFY
    auto taskInfo10 = InitTaskInfo(0, 10);
    taskInfo10->taskParam_.taskType = TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY;
    taskInfo10->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo10));
    // 加入当前异常 Task
    auto curTaskInfo = InitTaskInfo(0, 11); // streamId 0, taskId 11
    curTaskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    curTaskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    curTaskInfo->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(curTaskInfo));

    MOCKER(TaskExceptionHandler::PrintAicpuErrorMessage).stubs();
    // 调用 TaskExceptionHandler::Process() 打印异常DFX信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 11;  // 当前异常TaskId
    TaskExceptionHandler::Process(&exceptionInfo);

    globalMirrorTasks.DestroyQueue(0, 0);   // diveceId 0, streamId 0
}

HcclResult MockGetCcuErrorMsg(s32 deviceId, uint16_t missionStatus, uint16_t currIns, const ParaCcu &ccuTaskParam, std::vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo loopGroupErrorInfo{};
    loopGroupErrorInfo.type = CcuErrorType::LOOP_GROUP;
    loopGroupErrorInfo.repType = CcuRepType::LOOPGROUP;
    loopGroupErrorInfo.instrId = 1;
    loopGroupErrorInfo.msg.loopGroup.startLoopInsId = 17;
    loopGroupErrorInfo.msg.loopGroup.loopInsCnt = 5;
    loopGroupErrorInfo.msg.loopGroup.expandOffset = 3;
    loopGroupErrorInfo.msg.loopGroup.expandCnt = 2;
    errorInfo.push_back(loopGroupErrorInfo);

    CcuErrorInfo loopErrorInfo{};
    loopErrorInfo.type = CcuErrorType::LOOP;
    loopErrorInfo.repType = CcuRepType::LOOP;
    loopErrorInfo.instrId = 2;
    loopErrorInfo.msg.loop.startInstrId = 7;
    loopErrorInfo.msg.loop.endInstrId = 17;
    loopErrorInfo.msg.loop.loopCnt = 10;
    loopErrorInfo.msg.loop.loopCurrentCnt = 8;
    loopErrorInfo.msg.loop.addrStride = 0xaabbcc;
    errorInfo.push_back(loopErrorInfo);

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::LOC_WAIT_SEM;
    ccuErrorInfo.instrId = 3;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalValue = 0xabc;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;
    errorInfo.push_back(ccuErrorInfo);

    return HcclResult::HCCL_SUCCESS;
}

TEST_F(TaskExceptionHandlerTest, test_process_ccu)
{
    // 打桩 GlobalMirrorTasks
    GlobalMirrorTasks &globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);  // diveceId 0
    shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
    dfxOpInfo->commIndex_ = 3;
    dfxOpInfo->op_.dataCount = 0xff;
    dfxOpInfo->op_.reduceOp = ReduceOp::PROD;
    dfxOpInfo->op_.dataType = DataType::FP64;
    dfxOpInfo->algType_ = AlgType{AlgType::RING}.Describe();
    dfxOpInfo->op_.inputMem = make_shared<Buffer>(0x111122223333, 0);
    dfxOpInfo->op_.outputMem = make_shared<Buffer>(0xaaaabbbbcccc, 0);
    CommunicatorImpl communicator{};    // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 4;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    // 加入当前异常 Task
    auto curTaskInfo = InitTaskInfo(0, 0); // streamId 0, taskId 0
    curTaskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    curTaskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    mirrorTaskManager.AddTaskInfo(std::move(curTaskInfo));

    MOCKER(GetCcuErrorMsg).stubs().will(invoke(MockGetCcuErrorMsg));

    // 打桩清除TaskKill状态, 清除表项, 清除CKE操作
    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER(HrtGetDevicePhyIdByIndex).stubs().will(returnValue(static_cast<DevId>(0)));
    MOCKER(HrtRaCustomChannel).stubs();

    // 调用 TaskExceptionHandler::Process() 打印异常DFX信息
    MOCKER(TaskExceptionHandler::PrintAicpuErrorMessage).stubs();
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;  // 当前异常TaskId 0
    exceptionInfo.expandInfo.u.ccuInfo.ccuMissionNum = 1;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].dieId = 1;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].missionId = 2;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].instrId = 3;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].status = 4;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].subStatus = 5;
    TaskExceptionHandler::Process(&exceptionInfo);

    globalMirrorTasks.DestroyQueue(0, 0);   // diveceId 0, streamId 0
}

TEST_F(TaskExceptionHandlerTest, test_GetMC2AlgTaskParam)
{
    auto taskInfo = InitTaskInfo();

    taskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    EXPECT_EQ(TaskExceptionHandler::GetMC2AlgTaskParam(*taskInfo).size(), 0);    // failed

    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo->dfxOpInfo_->comm_ = nullptr;
    EXPECT_EQ(TaskExceptionHandler::GetMC2AlgTaskParam(*taskInfo).size(), 0);    // failed

    // Mock CommunicatorImpl
    CommunicatorImpl communicator{};
    communicator.collServices[AcceleratorState::CCU_SCHED] = std::make_shared<CollServiceDeviceMode>(&communicator);
    taskInfo->dfxOpInfo_->comm_ = &communicator;
    EXPECT_EQ(TaskExceptionHandler::GetMC2AlgTaskParam(*taskInfo).size(), 0);    // failed

    // Mock collService
    communicator.collService = new CollServiceDeviceMode(&communicator);
    auto* collServiceCcu = static_cast<CollServiceDeviceMode*>(communicator.GetCcuCollService());
    collServiceCcu->mc2Compont.ccuServerMap[10] = {1};
    CcuTaskParam ccuTaskParam{};
    std::vector<std::vector<CcuTaskParam>> ccuTaskParams{};
    ccuTaskParams.push_back({ccuTaskParam});
    ccuTaskParams.push_back({ccuTaskParam});
    collServiceCcu->mc2Compont.algoTemplateMap[1] = ccuTaskParams;
    taskInfo->taskParam_.taskPara.Ccu.executeId = 10;
    EXPECT_EQ(TaskExceptionHandler::GetMC2AlgTaskParam(*taskInfo).size(), 2);   // success
    delete communicator.collService;
}

TEST_F(TaskExceptionHandlerTest, test_process_mc2)
{
    auto taskInfo1 = InitTaskInfo();    // for MC2 Server
    taskInfo1->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo1->taskParam_.taskPara.Ccu.dieId = 0;
    taskInfo1->taskParam_.taskPara.Ccu.missionId = 1;
    taskInfo1->taskParam_.taskPara.Ccu.instrId = 2;
    MC2GlobalMirrorTasks::GetInstance().AddTaskInfo(10, std::shared_ptr<TaskInfo>(std::move(taskInfo1)));
    auto taskInfo2 = InitTaskInfo();    // for Algo
    taskInfo2->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo2->taskParam_.taskPara.Ccu.dieId = 0;
    taskInfo2->taskParam_.taskPara.Ccu.missionId = 2;
    taskInfo2->taskParam_.taskPara.Ccu.instrId = 5;
    MC2GlobalMirrorTasks::GetInstance().AddTaskInfo(10, std::shared_ptr<TaskInfo>(std::move(taskInfo2)));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.expandInfo.type = RT_EXCEPTION_FUSION;
    exceptionInfo.expandInfo.u.fusionInfo.type = RT_FUSION_AICORE_CCU;
    exceptionInfo.deviceid = 10;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.ccuMissionNum = 1;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].dieId = 0;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].missionId = 1;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].instrId = 2;

    MOCKER(GetCcuErrorMsg).stubs().will(returnValue(HcclResult::HCCL_SUCCESS)).then(invoke(MockGetCcuErrorMsg));
    CcuTaskParam ccuTaskParam{};
    ccuTaskParam.dieId = 0;
    ccuTaskParam.missionId = 2;
    ccuTaskParam.instStartId = 5;
    vector<CcuTaskParam> mockAlgTaskParams{ccuTaskParam};
    MOCKER(TaskExceptionHandler::GetMC2AlgTaskParam).stubs().will(returnValue(mockAlgTaskParams));

    // 打桩清除TaskKill状态, 清除表项, 清除CKE操作
    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER(HrtGetDevicePhyIdByIndex).stubs().will(returnValue(0));
    MOCKER(HrtRaCustomChannel).stubs();

    TaskExceptionHandler::Process(&exceptionInfo);
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessAivException_When_Normal_Expect_PrintInfo)
{
    // 初始化AIV任务信息
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_AIV;
    taskInfo->taskParam_.taskPara.Aiv.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    taskInfo->taskParam_.taskPara.Aiv.tag = 2;
    taskInfo->taskParam_.taskPara.Aiv.rank = 3;
    taskInfo->taskParam_.taskPara.Aiv.rankSize = 4;
    taskInfo->taskParam_.taskPara.Aiv.count = 5;
    taskInfo->taskParam_.taskPara.Aiv.numBlocks = 6;
    taskInfo->taskParam_.taskPara.Aiv.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    taskInfo->taskParam_.beginTime = 123456789;
    
    // 模拟flag内存
    int32_t flagMemData[] = {10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0};
    size_t flagMemSize = sizeof(flagMemData);
    taskInfo->taskParam_.taskPara.Aiv.flagMem = flagMemData;
    taskInfo->taskParam_.taskPara.Aiv.flagMemSize = flagMemSize;

    // 打桩ACL函数
    void* mockFlagBuff = malloc(flagMemSize);
    MOCKER(aclrtMallocHost).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtFreeHost).stubs().will(returnValue(ACL_SUCCESS));

    // 打桩PrintAivPreviousTaskException
    MOCKER(TaskExceptionHandler::PrintAivPreviousTaskException).stubs();

    // 构造异常信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    // 调用ProcessAivException（必须在AddTaskInfo之前，因为AddTaskInfo会move走taskInfo）
    TaskExceptionHandler handler(0);
    handler.ProcessAivException(&exceptionInfo, *taskInfo);

    // 打桩GlobalMirrorTasks
    GlobalMirrorTasks &globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    // 清理
    globalMirrorTasks.DestroyQueue(0, 0);
    if (mockFlagBuff) {
        free(mockFlagBuff);
    }
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessAivException_When_MallocFailure_Expect_ReturnEarly)
{
    // 初始化AIV任务信息
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_AIV;
    taskInfo->taskParam_.taskPara.Aiv.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    taskInfo->taskParam_.taskPara.Aiv.flagMemSize = 1024;

    // 打桩aclrtMallocHost失败
    MOCKER(aclrtMallocHost).stubs().will(returnValue(ACL_ERROR_BAD_ALLOC));

    // 构造异常信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    // 调用ProcessAivException（必须在AddTaskInfo之前，因为AddTaskInfo会move走taskInfo）
    TaskExceptionHandler handler(0);
    handler.ProcessAivException(&exceptionInfo, *taskInfo);

    // 打桩GlobalMirrorTasks
    GlobalMirrorTasks &globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    // 清理
    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessAivException_When_MemcpyFailure_Expect_ReturnEarly)
{
    // 初始化AIV任务信息
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_AIV;
    taskInfo->taskParam_.taskPara.Aiv.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    taskInfo->taskParam_.taskPara.Aiv.flagMemSize = 1024;

    // 打桩aclrtMallocHost成功，但aclrtMemcpy失败
    MOCKER(aclrtMallocHost).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_ERROR_RT_MEMORY_FREE));
    MOCKER(aclrtFreeHost).stubs().will(returnValue(ACL_SUCCESS));

    // 构造异常信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    // 调用ProcessAivException（必须在AddTaskInfo之前，因为AddTaskInfo会move走taskInfo）
    TaskExceptionHandler handler(0);
    handler.ProcessAivException(&exceptionInfo, *taskInfo);

    // 打桩GlobalMirrorTasks
    GlobalMirrorTasks &globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    // 清理
    globalMirrorTasks.DestroyQueue(0, 0);
}