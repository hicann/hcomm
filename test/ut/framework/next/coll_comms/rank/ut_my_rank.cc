/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "rank_graph_interface.h"
#include "rank_graph_v2.h"
#include "hccl/hccl_res.h"
#include <hccl/hccl_comm.h>
#include "hcomm_c_adpt.h"
#include "channel_process.h"
#include "base_config.h"
#include "ccu_device_res.h"
#include "config/env_config.h"
#include "env_config/env_config_v2.h"
#define private public
#include "my_rank.h"
#undef private
#include "hccl_comm_pub.h"
#include "llt_hccl_stub_rank_graph.h"
#include "ccu_res.h"

using namespace hccl;

HcclResult GetAllMemoryStub(CommMems*, std::vector<HcclMem>& mems, std::vector<std::string>& tags, uint64_t& version)
{
    HcclMem m;
    m.addr = (void*)0xCC11000;
    m.size = 1024;
    m.type = HCCL_MEM_TYPE_DEVICE;
    mems.push_back(m);
    tags.push_back("HcclBuffer");
    version = 1;
    return HCCL_SUCCESS;
}
HcclResult RegisterMemoryStub(
    hcomm::EndpointMgr*, EndpointHandle, const std::vector<std::string>&, const std::vector<HcclMem>& mems, uint64_t)
{
    return HCCL_SUCCESS;
}
HcclResult GetMemHandlesByTagsStub(
    hcomm::EndpointMgr* mgr, EndpointHandle, const std::vector<std::string>& tags, std::vector<MemHandle>& memHandleVec)
{
    for (size_t i = 0; i < tags.size(); i++) {
        memHandleVec.push_back((MemHandle)(0xCC11000 + i));
    }
    return HCCL_SUCCESS;
}

// spy 桩：记录 RegisterMemory 调用信息
struct RegisterMemoryCallRecord {
    EndpointHandle ep;
    std::vector<std::string> tags;
    std::vector<HcclMem> mems;
};
static std::vector<RegisterMemoryCallRecord> s_registerMemoryCalls;
HcclResult RegisterMemorySpyStub(
    hcomm::EndpointMgr* mgr, EndpointHandle ep, const std::vector<std::string>& tags, const std::vector<HcclMem>& mems,
    uint64_t version)
{
    // 模拟真实 RegisterMemory：版本一致则跳过
    auto [it, _] = mgr->endpointTagMemMap_.try_emplace(ep, ep);
    if (it->second.GetVersion() == version) {
        return HCCL_SUCCESS;
    }
    s_registerMemoryCalls.push_back({ep, tags, mems});
    it->second.SetVersion(version);
    return HCCL_SUCCESS;
}

// E2E 测试用的 spy 桩：统计 HcommMemReg/HcommMemUnreg 调用
static int s_e2eRegCount = 0;
static int s_e2eUnregCount = 0;
static HcommResult E2EMemRegStub(EndpointHandle, const char* tag, const CommMem*, HcommMemHandle* h)
{
    s_e2eRegCount++;
    *h = (HcommMemHandle)(uintptr_t)(0xCC100000 + s_e2eRegCount);
    return HCCL_SUCCESS;
}

static HcommResult E2EMemUnregStub(EndpointHandle, HcommMemHandle)
{
    s_e2eUnregCount++;
    return HCCL_SUCCESS;
}

static HcommResult E2EEndpointDestroyStub(EndpointHandle) { return HCCL_SUCCESS; }

class MyRankTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MyRankTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "MyRankTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        s_registerMemoryCalls.clear();
        std::cout << "A Test case in MyRankTest SetUP" << std::endl;
        rankIpPortMap = std::make_shared<std::unordered_map<u32, std::unordered_map<Hccl::IpAddress, u32>>>();
        (*rankIpPortMap)[0][Hccl::IpAddress("1.0.0.0")] = 16666;
        (*rankIpPortMap)[1][Hccl::IpAddress("2.0.0.0")] = 16666;
        (*rankIpPortMap)[2][Hccl::IpAddress("0.0.0.0")] = 16666;
        rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
        myRank = std::make_unique<MyRank>(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in MyRankTest TearDown" << std::endl;
    }

    void CreateCclBuffer(HcclMem& cclBuffer)
    {
        cclBuffer.addr = (void*)0xab;
        cclBuffer.size = 1024;
        cclBuffer.type = HCCL_MEM_TYPE_DEVICE;
    }

    void CreateEndpointDesc(EndpointDesc& ep, CommProtocol protocol, const std::string& ip)
    {
        ep.protocol = protocol;
        ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ep.commAddr.addr = Hccl::IpAddress(ip).GetBinaryAddress().addr;
        ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    }

    void MockerFuncs()
    {
        MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue((Hccl::Socket*)0xab));
        MOCKER_CPP(&hccl::CommMems::GetAllMemory).stubs().will(invoke(GetAllMemoryStub));
        MOCKER_CPP(&hcomm::EndpointMgr::RegisterMemory).stubs().will(invoke(RegisterMemoryStub));
        MOCKER_CPP(&hcomm::EndpointMgr::GetMemHandlesByTags).stubs().will(invoke(GetMemHandlesByTagsStub));
        MOCKER(HcommCcuInsCreate).stubs().with(mockcpp::any()).will(returnValue(CcuResult::CCU_SUCCESS));
        MOCKER(HcommCcuInsCreateLegacy).stubs().with(mockcpp::any()).will(returnValue(CcuResult::CCU_SUCCESS));
        MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).stubs().will(returnValue(HCCL_SUCCESS));
    }

    uint32_t DEFAULT_MODE = 0;
    uint32_t AICPU_TS_MODE = 2;
    uint32_t CCU_MS_MODE = 5;
    uint32_t CCU_SCHED_MODE = 6;
    Hccl::RankIpPortMapPtr rankIpPortMap;
    aclrtBinHandle binHandle;
    CommConfig config{"my_rank_ut"};
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph;
    std::unique_ptr<MyRank> myRank;
};

TEST_F(MyRankTest, Ut_When_QueryListenPort_Listen_Port_Expect_SUCCESS)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "2.0.0.0");

    uint32_t listenPort;
    HcommChannelDesc desc;
    HcclResult ret = myRank->QueryListenPort(0, 1, localEp, rmtEp, listenPort, desc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listenPort, devPort);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_SERVER);

    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_ROCE, "0.0.0.0");
    ret = myRank->QueryListenPort(0, 2, localEp, rmtEp2, listenPort, desc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listenPort, devPort);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_CLIENT);
}

TEST_F(MyRankTest, Ut_When_QueryListenPort_InValid_Port_Expect_E_PARA)
{
    uint32_t devPort = 1919000;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "2.0.0.0");

    uint32_t listenPort;
    HcommChannelDesc desc;
    HcclResult ret = myRank->QueryListenPort(0, 1, localEp, rmtEp, listenPort, desc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(MyRankTest, Ut_When_BatchCreateChannels_Expect_SUCCESS)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();
    ChannelHandle channelHandle = 0xab;
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);
    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_MEM, "0.0.0.0");

    HcclChannelDesc channelDesc[3]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].ubMemAttr.pathMode = 0;
    channelDesc[1].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[1].remoteRank = 1;
    channelDesc[1].notifyNum = 2;
    channelDesc[1].localEndpoint = localEp;
    channelDesc[1].remoteEndpoint = rmtEp;
    channelDesc[1].ubMemAttr.pathMode = 0;
    channelDesc[2].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[2].remoteRank = 2;
    channelDesc[2].notifyNum = 2;
    channelDesc[2].localEndpoint = localEp;
    channelDesc[2].remoteEndpoint = rmtEp2;
    channelDesc[2].ubMemAttr.pathMode = 0;

    u32 channelIdx0 = 0u;
    u32 channelIdx1 = 1u;
    u32 channelIdx2 = 2u;
    u32 RmtEp1reuseIdx0 = 0u;
    u32 RmtEp1reuseIdx1 = 1u;
    u32 RmtEp2reuseIdx0 = 0u;

    std::vector<HcommChannelDesc> hcommDesc(3);
    for (u32 i = 0; i < 3; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 1, "test", hcommDesc), HCCL_SUCCESS);
    std::vector<ChannelHandle> hostChannelHandles(3);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    std::vector<std::vector<MemHandle>> allHandles1(1);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AICPU_TS, channelDesc, 1, hcommDesc, hostChannelHandleList, allHandles1),
        HCCL_SUCCESS);
    EXPECT_EQ(myRank->newChannels_.size(), 1);
    EXPECT_EQ(myRank->newChannels_[0], std::make_pair(channelIdx0, RmtEp1reuseIdx0));

    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 2, "test", hcommDesc), HCCL_SUCCESS);
    std::vector<std::vector<MemHandle>> allHandles2(2);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AICPU_TS, channelDesc, 2, hcommDesc, hostChannelHandleList, allHandles2),
        HCCL_SUCCESS);
    EXPECT_EQ(myRank->newChannels_.size(), 1);
    EXPECT_EQ(myRank->newChannels_[0], std::make_pair(channelIdx1, RmtEp1reuseIdx1));

    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 3, "test", hcommDesc), HCCL_SUCCESS);
    std::vector<std::vector<MemHandle>> allHandles3(3);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AICPU_TS, channelDesc, 3, hcommDesc, hostChannelHandleList, allHandles3),
        HCCL_SUCCESS);
    EXPECT_EQ(myRank->newChannels_.size(), 1);
    EXPECT_EQ(myRank->newChannels_[0], std::make_pair(channelIdx2, RmtEp2reuseIdx0));

    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_AGAIN));
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().with(mockcpp::any()).will(returnValue((s32)(1)));
    EXPECT_EQ(myRank->BatchConnectChannels(channelDesc, hostChannelHandleList, 3), HCCL_E_TIMEOUT);
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_TIMEOUT));
    EXPECT_EQ(myRank->BatchConnectChannels(channelDesc, hostChannelHandleList, 3), HCCL_E_TIMEOUT);
    unsetenv("HCCL_DFS_CONFIG");
}

// 测试Init时用户未配置展开模式时，读取环境变量配置
TEST_F(MyRankTest, Ut_Init_When_Default_Mode_Expect_Set_By_Env)
{
    setenv("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", 1);
    // EnvConfig为单例，首次GetInstance时构造并Parse一次；此处强制重新Parse使setenv生效
    Hccl::EnvConfig::GetInstance().Parse();
    MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t defaultOpExpansionMode = DEFAULT_MODE;
    EXPECT_EQ(myRank->Init(cclBuffer, defaultOpExpansionMode, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank->opExpansionMode_, CCU_SCHED_MODE);
    unsetenv("HCCL_OP_EXPANSION_MODE");
}

// 测试Init时ccu驱动拉起失败回退到aicpu
TEST_F(MyRankTest, Ut_Init_When_Ccu_Driver_Fail_Expect_Fallback_Aicpu)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(HcommCcuInsCreateLegacy).stubs().will(returnValue(CcuResult::CCU_E_DRV_BUSY));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank->Init(cclBuffer, opExpansionModeMs, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank->opExpansionMode_, AICPU_TS_MODE);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试Init时ccu ms资源不足回退到sched
TEST_F(MyRankTest, Ut_Init_When_Ccu_Ms_Insufficient_Expect_Fallback_Sched)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(HcommCcuInsCreateLegacy)
        .stubs()
        .will(returnValue(CcuResult::CCU_E_UNAVAIL))
        .then(returnValue(CcuResult::CCU_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank->Init(cclBuffer, opExpansionModeMs, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank->opExpansionMode_, CCU_SCHED_MODE);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试Init时ccu ms和sched资源不足回退到aicpu
TEST_F(MyRankTest, Ut_Init_When_Ccu_Ms_And_Sched_Insufficient_Expect_Fallback_Aicpu)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(HcommCcuInsCreateLegacy).stubs().will(returnValue(CcuResult::CCU_E_UNAVAIL));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank->Init(cclBuffer, opExpansionModeMs, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank->opExpansionMode_, AICPU_TS_MODE);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试Init在申请资源时出现其他报错时失败
TEST_F(MyRankTest, Ut_Init_When_Resource_Fail_Expect_Fail)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(HcommCcuInsCreateLegacy).stubs().will(returnValue(CcuResult::CCU_E_PARA));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank->Init(cclBuffer, opExpansionModeMs, 2), HCCL_E_PARA);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试BatchCreateChannels在资源不足时销毁新申请的channel
TEST_F(MyRankTest, St_BatchCreateChannels_When_Resource_fallback_Expect_Return_HCCL_E_UNAVAIL)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommEndpointStartListen).stubs().with(mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(HcommChannelDestroy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 5, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_CTP, "1.0.0.0");
    EndpointDesc rmtEp1;
    CreateEndpointDesc(rmtEp1, COMM_PROTOCOL_UB_CTP, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_CTP, "3.0.0.0");

    HcclChannelDesc channelDesc[5]{};
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp1;
    }
    for (int i = 2; i < 5; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 2;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp2;
    }

    // 模拟创建到rmtEp2的第二个channel时资源不足，需要清理前三个channel
    MOCKER(HcommCollectiveChannelCreate)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)))
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))
        .then(returnValue(static_cast<int>(HCCL_E_UNAVAIL)));
    std::vector<HcommChannelDesc> hcommDesc(5);
    std::vector<ChannelHandle> hostChannelHandles(5);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    std::vector<std::vector<MemHandle>> allHandles_fb1(5);
    EXPECT_EQ(
        myRank->BatchCreateChannels(COMM_ENGINE_CCU, channelDesc, 5, hcommDesc, hostChannelHandleList, allHandles_fb1),
        HCCL_E_UNAVAIL);
    EXPECT_EQ(myRank->newChannels_.size(), 0);

    // 获取到rmtEp1的endpointPair
    RankIdPair rankIdPair1 = std::make_pair(0, 1);
    EndpointDescPair endpointDescPair1 = std::make_pair(localEp, rmtEp1);
    RankPair* rankPair1 = nullptr;
    hcomm::EndpointPair* endpointPair1 = nullptr;
    myRank->rankPairMgr_->Get(rankIdPair1, rankPair1);
    rankPair1->GetEndpointPair(endpointDescPair1, endpointPair1);

    // 期望channelHandle被清理
    EXPECT_EQ(endpointPair1->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair1->channelHandles_.find(COMM_ENGINE_CCU), endpointPair1->channelHandles_.end());
    EXPECT_EQ(endpointPair1->channelHandles_[COMM_ENGINE_CCU].size(), 0);

    // 获取到rmtEp2的endpointPair
    RankIdPair rankIdPair2 = std::make_pair(0, 2);
    EndpointDescPair endpointDescPair2 = std::make_pair(localEp, rmtEp2);
    RankPair* rankPair2 = nullptr;
    hcomm::EndpointPair* endpointPair2 = nullptr;
    myRank->rankPairMgr_->Get(rankIdPair2, rankPair2);
    rankPair2->GetEndpointPair(endpointDescPair2, endpointPair2);

    // 期望channelHandle被清理
    EXPECT_EQ(endpointPair2->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair2->channelHandles_.find(COMM_ENGINE_CCU), endpointPair2->channelHandles_.end());
    EXPECT_EQ(endpointPair2->channelHandles_[COMM_ENGINE_CCU].size(), 0);
}

// 测试多次调用BatchCreateChannels，在最后一次资源不足时只销毁新申请的channel
TEST_F(MyRankTest, St_BatchCreateChannels_Multi_Times_When_fallback_Expect_Return_HCCL_E_UNAVAIL)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommEndpointStartListen).stubs().with(mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(HcommChannelDestroy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 5, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_CTP, "1.0.0.0");
    EndpointDesc rmtEp1;
    CreateEndpointDesc(rmtEp1, COMM_PROTOCOL_UB_CTP, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_CTP, "3.0.0.0");

    HcclChannelDesc channelDesc[5]{};
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp1;
    }
    for (int i = 2; i < 5; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 2;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp2;
    }

    // 第一次调用BatchCreateChannels，创建3个channel成功
    // 第二次调用BatchCreateChannels，创建5个channel，前4个channel成功，第5个channel失败
    // 需要只清理到rmtEp2的第二个channel
    MOCKER(HcommCollectiveChannelCreate)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)))    // 第一次调用，到rmtEp1的channel1成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))    // 第一次调用，到rmtEp1的channel2成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))    // 第一次调用，到rmtEp2的channel1成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS)))    // 第二次调用，到rmtEp2的channel2成功
        .then(returnValue(static_cast<int>(HCCL_E_UNAVAIL))); // 第二次调用，到rmtEp2的channel3失败
    std::vector<HcommChannelDesc> hcommDesc(5);
    std::vector<ChannelHandle> hostChannelHandles(5);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    // 第一次调用BatchCreateChannels成功，创建3个channel
    std::vector<std::vector<MemHandle>> allHandles_multi1(3);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_CCU, channelDesc, 3, hcommDesc, hostChannelHandleList, allHandles_multi1),
        HCCL_SUCCESS);
    EXPECT_EQ(myRank->newChannels_.size(), 3);
    u32 channelIdx0 = 0u;
    u32 channelIdx1 = 1u;
    u32 channelIdx2 = 2u;
    u32 RmtEp1reuseIdx0 = 0u;
    u32 RmtEp1reuseIdx1 = 1u;
    u32 RmtEp2reuseIdx0 = 0u;
    EXPECT_EQ(myRank->newChannels_[0], std::make_pair(channelIdx0, RmtEp1reuseIdx0));
    EXPECT_EQ(myRank->newChannels_[1], std::make_pair(channelIdx1, RmtEp1reuseIdx1));
    EXPECT_EQ(myRank->newChannels_[2], std::make_pair(channelIdx2, RmtEp2reuseIdx0));

    // 获取到rmtEp1的endpointPair
    RankIdPair rankIdPair1 = std::make_pair(0, 1);
    EndpointDescPair endpointDescPair1 = std::make_pair(localEp, rmtEp1);
    RankPair* rankPair1 = nullptr;
    hcomm::EndpointPair* endpointPair1 = nullptr;
    myRank->rankPairMgr_->Get(rankIdPair1, rankPair1);
    rankPair1->GetEndpointPair(endpointDescPair1, endpointPair1);

    // 获取到rmtEp2的endpointPair
    RankIdPair rankIdPair2 = std::make_pair(0, 2);
    EndpointDescPair endpointDescPair2 = std::make_pair(localEp, rmtEp2);
    RankPair* rankPair2 = nullptr;
    hcomm::EndpointPair* endpointPair2 = nullptr;
    myRank->rankPairMgr_->Get(rankIdPair2, rankPair2);
    rankPair2->GetEndpointPair(endpointDescPair2, endpointPair2);

    // 期望到rmtEp1的channelHandle有两个channel
    EXPECT_EQ(endpointPair1->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair1->channelHandles_.find(COMM_ENGINE_CCU), endpointPair1->channelHandles_.end());
    EXPECT_EQ(endpointPair1->channelHandles_[COMM_ENGINE_CCU].size(), 2);

    // 期望到rmtEp2的channelHandle有一个channel
    EXPECT_EQ(endpointPair2->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair2->channelHandles_.find(COMM_ENGINE_CCU), endpointPair2->channelHandles_.end());
    EXPECT_EQ(endpointPair2->channelHandles_[COMM_ENGINE_CCU].size(), 1);

    // 第二次调用BatchCreateChannels，创建第5个channel失败
    std::vector<std::vector<MemHandle>> allHandles_multi2(5);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_CCU, channelDesc, 5, hcommDesc, hostChannelHandleList, allHandles_multi2),
        HCCL_E_UNAVAIL);
    EXPECT_EQ(myRank->newChannels_.size(), 0);

    // 期望到rmtEp1的channelHandle不被清理，保持两个channel
    EXPECT_EQ(endpointPair1->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair1->channelHandles_.find(COMM_ENGINE_CCU), endpointPair1->channelHandles_.end());
    EXPECT_EQ(endpointPair1->channelHandles_[COMM_ENGINE_CCU].size(), 2);

    // 期望到rmtEp2的channelHandle只有第二次新创建的channel2被清理，保持第一次创建时的一个channel
    EXPECT_EQ(endpointPair2->channelHandles_.size(), 1);
    EXPECT_NE(endpointPair2->channelHandles_.find(COMM_ENGINE_CCU), endpointPair2->channelHandles_.end());
    EXPECT_EQ(endpointPair2->channelHandles_[COMM_ENGINE_CCU].size(), 1);
}

TEST_F(MyRankTest, Ut_When_ChannelGetHcclBuffer_NoBuffer_Expect_HCCL_E_INTERNAL)
{
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems).stubs().will(returnValue(HCCL_SUCCESS));

    ChannelHandle channel = 0x12345;
    void* buffer = nullptr;
    uint64_t size = 0;
    HcclResult ret = myRank->ChannelGetHcclBuffer(channel, &buffer, &size);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_Normal_Expect_SUCCESS)
{
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems).stubs().will(returnValue(HCCL_SUCCESS));

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    char** memInfo = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, &remoteMem, &memInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_RemoteMemNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    char** memInfo = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, nullptr, &memInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_MemInfoNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, &remoteMem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_MemNumNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    char** memInfo = nullptr;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, nullptr, &remoteMem, &memInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMemsVec_Normal_Expect_SUCCESS)
{
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems).stubs().will(returnValue(HCCL_SUCCESS));

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    std::vector<std::string> memTags;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, &remoteMem, memTags);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMemsVec_RemoteMemNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    uint32_t memNum = 0;
    std::vector<std::string> memTags;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, nullptr, memTags);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMemsVec_MemNumNull_Expect_E_PTR)
{
    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    std::vector<std::string> memTags;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, nullptr, &remoteMem, memTags);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMemsVec_MultiTagsWithNull_Expect_CopiedToVector)
{
    // 验证 memNum>0 底层返回多个 memTag（含 nullptr）时，char** 能否被正确拷贝为 vector<string>（nullptr 转空串）
    CommMem remoteMems[2] = {};
    char tag0[] = "ut_remote_tag";
    char* rawTags[2] = {tag0, nullptr};
    uint32_t expectMemNum = 2;
    CommMem* expectRemoteMem = remoteMems;
    char** expectTags = rawTags;
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems)
        .stubs()
        .with(mockcpp::any(), outBoundP(&expectMemNum), outBoundP(&expectRemoteMem), outBoundP(&expectTags))
        .will(returnValue(HCCL_SUCCESS));

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    std::vector<std::string> memTags;
    HcclResult ret = myRank->ChannelGetRemoteMems(channel, &memNum, &remoteMem, memTags);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(memNum, 2U);
    ASSERT_EQ(memTags.size(), 2U);
    EXPECT_EQ(memTags[0], std::string("ut_remote_tag"));
    EXPECT_EQ(memTags[1], std::string());
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcCtp_Sets_CommDomainQos_FromCommConfig)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(5u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_CTP;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 5u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcTp_Sets_CommDomainQos_FromCommConfig)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(2u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_TP;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 2u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_Uboe_Sets_CommDomainQos_FromCommConfig)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(7u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBOE;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 7u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcTp_QosUnset_UsesUbQosDefault)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(HCCL_COMM_QOS_CONFIG_NOT_SET), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_TP;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 4u); // 与 EnvConfig::UB_QOS_DEFAULT 一致
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_Roce_DoesNotUseUbAttrBranch)
{
    CommConfig commConfig("ut");
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_ROCE;
    in.roceAttr.retryCnt = 3u;
    in.roceAttr.retryInterval = 20u;
    in.roceAttr.tc = 8u;
    in.roceAttr.sl = 4u;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.roceAttr.retryCnt, 3u);
    EXPECT_EQ(out.roceAttr.retryInterval, 20u);
    EXPECT_EQ(out.roceAttr.tc, 8u);
    EXPECT_EQ(out.roceAttr.sl, 4u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbMem_Sets_PathMode_FromInput)
{
    CommConfig commConfig("ut");
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_MEM;
    in.ubMemAttr.pathMode = 2u;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.ubMemAttr.pathMode, 2u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbMem_PathModeZero_PassedThrough)
{
    CommConfig commConfig("ut");
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_MEM;
    in.ubMemAttr.pathMode = 0u;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.ubMemAttr.pathMode, 0u);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcCtp_DoesNotSetPathMode)
{
    // UBC_CTP 走 qos 分支，不进入 UB_MEM 的 pathMode 显式设置分支；
    // pathMode 仅经 union raws 拷贝传递，这里验证 ubMemAttr.pathMode 仍随 raws 透传
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(5u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_CTP;
    in.ubMemAttr.pathMode = 2u;
    HcommChannelDesc out = MyRankUtils::ChannelDescHccl2Hcomm(in, commConfig);
    EXPECT_EQ(out.qos, 5u);
    EXPECT_EQ(out.ubMemAttr.pathMode, 2u);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CCU_MSModel_WithCommConfig)
{
    HcommChannelDesc in{};
    ASSERT_EQ(myRank->config_.SetConfigSqDepth(512U), HCCL_SUCCESS);
    myRank->opExpansionMode_ = CCU_MS_MODE;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, in);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(in.ubAttr.sqDepth, 128);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CCU_SCHEDModel_WithCommConfig)
{
    HcommChannelDesc in{};
    ASSERT_EQ(myRank->config_.SetConfigSqDepth(512U), HCCL_SUCCESS);
    myRank->opExpansionMode_ = CCU_SCHED_MODE;
    HcclResult out = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, in);
    EXPECT_EQ(out, HCCL_SUCCESS);
    EXPECT_EQ(in.ubAttr.sqDepth, 16);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_EngineAndProtocolVary_Expect_ScopedPropagation)
{
    struct TestCase {
        CommEngine engine;
        CommProtocol protocol;
        uint32_t configuredSqDepth;
        uint32_t initialSqDepth;
        uint32_t expectedSqDepth;
    };

    const TestCase testCases[] = {
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UBC_TP, 128U, 64U, 128U},
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UB_CTP, 128U, 64U, 128U},
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UB_RTP, 128U, 64U, 128U},
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UBC_TP, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET,
         HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET},
        {COMM_ENGINE_AIV, COMM_PROTOCOL_UBOE, 128U, 64U, 64U},
        {COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UBC_TP, 128U, 64U, 64U},
    };

    for (const auto& testCase : testCases) {
        SCOPED_TRACE(
            testing::Message() << "engine=" << testCase.engine << ", protocol=" << testCase.protocol
                               << ", configuredSqDepth=" << testCase.configuredSqDepth);
        ASSERT_EQ(myRank->config_.SetConfigSqDepth(testCase.configuredSqDepth), HCCL_SUCCESS);
        HcommChannelDesc channelDesc{};
        channelDesc.remoteEndpoint.protocol = testCase.protocol;
        channelDesc.ubAttr.sqDepth = testCase.initialSqDepth;

        EXPECT_EQ(myRank->ConfigSqDepthByExpansionMode(testCase.engine, channelDesc), HCCL_SUCCESS);
        EXPECT_EQ(channelDesc.ubAttr.sqDepth, testCase.expectedSqDepth);
    }
}

// 测试TryInitCcuInstance在DEFAULT_MODE(0)时映射为CCU_UNUSED，提前返回成功且不拉起CCU
// 本次提交将DEFAULT_MODE从CCU_SCHED改为CCU_UNUSED，此处验证新行为
TEST_F(MyRankTest, Ut_TryInitCcuInstance_When_DefaultMode_Expect_CcuUnusedAndSuccess)
{
    myRank->opExpansionMode_ = DEFAULT_MODE;
    HcclResult ret = myRank->TryInitCcuInstance();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank->ccuInsHandle_, static_cast<CcuInsHandle>(0));
}

// 测试TryInitCcuInstance在AICPU_TS_MODE(2)时映射为CCU_UNUSED，提前返回成功且不拉起CCU
TEST_F(MyRankTest, Ut_TryInitCcuInstance_When_AicpuTsMode_Expect_CcuUnusedAndSuccess)
{
    myRank->opExpansionMode_ = AICPU_TS_MODE;
    HcclResult ret = myRank->TryInitCcuInstance();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank->ccuInsHandle_, static_cast<CcuInsHandle>(0));
}

// 测试HCCL_OP_EXPANSION_MODE未配置时默认值为AICPU_TS
// 本次提交将env_config默认值由CCU_SCHED改为AICPU_TS，此处直接构造EnvAlgoConfig验证新默认值
TEST_F(MyRankTest, Ut_EnvAlgoConfig_When_NoExpansionModeEnv_Expect_DefaultAicpuTs)
{
    unsetenv("HCCL_OP_EXPANSION_MODE");
    Hccl::EnvAlgoConfig algoConfig;
    algoConfig.Parse();
    EXPECT_EQ(algoConfig.GetHcclAccelerator(), Hccl::HcclAccelerator::AICPU_TS);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_BatchExchangeAndCheckConsistency_Timeout_Expect_HCCL_E_TIMEOUT)
{
    MOCKER_CPP(&hccl::MyRank::BatchCreateSockets).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::MyRank::BatchCreateChannels).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MockerFuncs();

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::ExchangeInfoMgr::BatchExchangeAndCheckConsistency)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_TIMEOUT));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    HcclChannelDesc channelDesc[1]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].ubMemAttr.pathMode = 0;

    ChannelHandle channelHandles[1];
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", channelDesc, 1, channelHandles);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_NullChannelDesc_Expect_EPtr)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    ChannelHandle channelHandles[1];
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", nullptr, 1, channelHandles);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_NullChannelHandles_Expect_EPtr)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    HcclChannelDesc channelDesc[1];
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", channelDesc, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_CreateChannels_When_ZeroChannelNum_Expect_EPara)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    HcclChannelDesc channelDesc[1];
    ChannelHandle channelHandles[1];
    HcclResult ret = myRank->CreateChannels(COMM_ENGINE_AICPU_TS, "test", channelDesc, 0, channelHandles);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CcuMs_Expect_CorrectDepth)
{
    myRank->opExpansionMode_ = CCU_MS_MODE;
    HcommChannelDesc hcommDesc;
    hcommDesc.ubAttr.sqDepth = 0;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, hcommDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcommDesc.ubAttr.sqDepth, 128u);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CcuSched_Expect_CorrectDepth)
{
    myRank->opExpansionMode_ = CCU_SCHED_MODE;
    HcommChannelDesc hcommDesc;
    hcommDesc.ubAttr.sqDepth = 0;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, hcommDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcommDesc.ubAttr.sqDepth, 16u);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CcuInvalidMode_Expect_EInternal)
{
    myRank->opExpansionMode_ = DEFAULT_MODE;
    HcommChannelDesc hcommDesc;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, hcommDesc);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_NonCcuEngine_Expect_Success)
{
    myRank->opExpansionMode_ = DEFAULT_MODE;
    HcommChannelDesc hcommDesc;
    HcclResult ret = myRank->ConfigSqDepthByExpansionMode(COMM_ENGINE_AICPU_TS, hcommDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试重构前后 hcommDescs 输出等价：老代码(GetTagMemoryHandles+RegisterMemory)
// 和新代码(GetAllMemory+PrepareChannelMemHandles) 输出一致
TEST_F(MyRankTest, Ut_BatchCreateChannels_HcommDescsEquivalence)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    // 独立mock: 不走MockerFuncs(), 让GetAllMemory+RegisterMemory跑真实代码
    // COMM_PROTOCOL_UB_MEM → UbMemEndpoint → UbMemRegedMemMgr, 纯内存操作无需硬件资源
    MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue((Hccl::Socket*)0xab));
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    // 注册两块用户内存，拿到真实 rawHandle
    CommMem memA{};
    memA.addr = (void*)0xA000;
    memA.size = 4096;
    memA.type = COMM_MEM_TYPE_DEVICE;
    void* rawA = nullptr;
    EXPECT_EQ(myRank->commMems_->CommRegMem("tagA", memA, &rawA), HCCL_SUCCESS);
    ASSERT_NE(rawA, nullptr);

    CommMem memB{};
    memB.addr = (void*)0xB000;
    memB.size = 8192;
    memB.type = COMM_MEM_TYPE_DEVICE;
    void* rawB = nullptr;
    EXPECT_EQ(myRank->commMems_->CommRegMem("tagB", memB, &rawB), HCCL_SUCCESS);
    ASSERT_NE(rawB, nullptr);

    // 2 endpoint, 4 channel，用户 mem 子集各不相同
    EndpointDesc epA, epB;
    CreateEndpointDesc(epA, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    CreateEndpointDesc(epB, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "3.0.0.0");

    HcclChannelDesc channelDesc[4]{};
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = epA;
        channelDesc[i].remoteEndpoint = rmtEp;
    }
    for (int i = 2; i < 4; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
        channelDesc[i].remoteRank = 2;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = epB;
        channelDesc[i].remoteEndpoint = rmtEp;
    }
    // ch0: epA, [tagA]
    channelDesc[0].memHandles = &rawA;
    channelDesc[0].memHandleNum = 1;
    // ch1: epA, [tagB]
    channelDesc[1].memHandles = &rawB;
    channelDesc[1].memHandleNum = 1;
    // ch2: epB, [tagA, tagB]
    void* handlesAB[] = {rawA, rawB};
    channelDesc[2].memHandles = handlesAB;
    channelDesc[2].memHandleNum = 2;
    // ch3: epB, memHandleNum=0（仅 cclBuffer）
    channelDesc[3].memHandles = nullptr;
    channelDesc[3].memHandleNum = 0;

    std::vector<HcommChannelDesc> hcommDesc(4);
    for (u32 i = 0; i < 4; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 4, "test", hcommDesc), HCCL_SUCCESS);

    std::vector<ChannelHandle> hostChannelHandles(4);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    std::vector<std::vector<MemHandle>> allHandles_eq(4);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AICPU_TS, channelDesc, 4, hcommDesc, hostChannelHandleList, allHandles_eq),
        HCCL_SUCCESS);

    // 验证 hcommDescs：输出仅取决于 channelDesc 的 memHandle 子集，与内部注册方式无关
    ASSERT_EQ(hcommDesc[0].memHandleNum, 2u); // cclBuffer + tagA
    ASSERT_EQ(hcommDesc[1].memHandleNum, 2u); // cclBuffer + tagB
    ASSERT_EQ(hcommDesc[2].memHandleNum, 3u); // cclBuffer + tagA + tagB
    ASSERT_EQ(hcommDesc[3].memHandleNum, 1u); // 仅 cclBuffer
    EXPECT_EQ(hcommDesc[0].exchangeAllMems, false);
    EXPECT_EQ(hcommDesc[1].exchangeAllMems, false);
    EXPECT_EQ(hcommDesc[2].exchangeAllMems, false);
    EXPECT_EQ(hcommDesc[3].exchangeAllMems, false);

    unsetenv("HCCL_DFS_CONFIG");
}

// 测试 PrepareChannelMemHandles 在 RegisterMemory 失败时直接返回错误
TEST_F(MyRankTest, Ut_PrepareChannelMemHandles_RegisterMemoryFail)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    MOCKER_CPP(&hcomm::EndpointMgr::RegisterMemory).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    EndpointHandle epHandle = (EndpointHandle)0x1;
    std::vector<MemHandle> memHandleVec;
    HcclResult ret = myRank->PrepareMemHandles(epHandle, nullptr, 0, memHandleVec);

    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_TRUE(memHandleVec.empty());
}

// 测试 PrepareChannelMemHandles 在 memHandles==nullptr 但 memHandleNum>0 时返回错误
TEST_F(MyRankTest, Ut_PrepareChannelMemHandles_NullMemHandles)
{
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    MockerFuncs();
    EndpointHandle epHandle = (EndpointHandle)0x1;

    // 正常：memHandleNum=0
    std::vector<MemHandle> handlesOk;
    EXPECT_EQ(myRank->PrepareMemHandles(epHandle, nullptr, 0, handlesOk), HCCL_SUCCESS);

    // 异常：memHandleNum>0 但 memHandles==nullptr，cclBuffer 不受影响
    std::vector<MemHandle> handlesBad;
    HcclResult ret = myRank->PrepareMemHandles(epHandle, nullptr, 3, handlesBad);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(handlesBad.size(), 1u); // cclBuffer 不因用户内存异常而丢失
}

// 验证 RegisterMemory 调用粒度：重构前 per-channel，重构后 per-endpoint
// 4 channel 共享 2 个 endpoint → RegisterMemory 只应调用 2 次
TEST_F(MyRankTest, Ut_BatchCreateChannels_RegisterMemoryPerEndpoint)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue((Hccl::Socket*)0xab));
    MOCKER_CPP(&hccl::CommMems::GetAllMemory).stubs().will(invoke(GetAllMemoryStub));
    MOCKER_CPP(&hcomm::EndpointMgr::RegisterMemory).stubs().will(invoke(RegisterMemorySpyStub));
    MOCKER_CPP(&hcomm::EndpointMgr::GetMemHandlesByTags).stubs().will(invoke(GetMemHandlesByTagsStub));
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    // 2 endpoint, 4 channel
    EndpointDesc epA, epB;
    CreateEndpointDesc(epA, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    CreateEndpointDesc(epB, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "3.0.0.0");

    HcclChannelDesc channelDesc[4]{};
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_MEM;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = epA;
        channelDesc[i].remoteEndpoint = rmtEp;
    }
    for (int i = 2; i < 4; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UB_MEM;
        channelDesc[i].remoteRank = 2;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = epB;
        channelDesc[i].remoteEndpoint = rmtEp;
    }

    std::vector<HcommChannelDesc> hcommDesc(4);
    for (u32 i = 0; i < 4; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 4, "test", hcommDesc), HCCL_SUCCESS);

    std::vector<ChannelHandle> hostChannelHandles(4);
    std::vector<std::vector<MemHandle>> allHandles(4);
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();
    EXPECT_EQ(
        myRank->BatchCreateChannels(COMM_ENGINE_AICPU_TS, channelDesc, 4, hcommDesc, hostChannelHandleList, allHandles),
        HCCL_SUCCESS);

    // 验证：2个endpoint → 2次RegisterMemory调用（非4次per-channel）
    ASSERT_EQ(s_registerMemoryCalls.size(), 2u);
    // 每次调用传入的是同一份全量数据（GetAllMemoryStub返回的1条cclBuffer）
    EXPECT_EQ(s_registerMemoryCalls[0].tags.size(), 1u);
    EXPECT_EQ(s_registerMemoryCalls[0].tags[0], "HcclBuffer");
    EXPECT_EQ(s_registerMemoryCalls[1].tags.size(), 1u);
    EXPECT_EQ(s_registerMemoryCalls[1].tags[0], "HcclBuffer");

    unsetenv("HCCL_DFS_CONFIG");
}

// 端到端生命周期：CommRegMem → CreateChannels → CommUnregMem → CreateChannels
// 验证解注册推送到全部endpoint，重进建链不含旧tag
TEST_F(MyRankTest, Ut_MemRegAndAcquireLifecycle)
{
    // ---- mock ----
    s_e2eRegCount = 0;
    s_e2eUnregCount = 0;

    // spy: HcommMemReg/Unreg 计数
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(E2EMemRegStub));
    MOCKER(HcommMemUnreg).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(E2EMemUnregStub));
    MOCKER(HcommEndpointDestroy).stubs().with(mockcpp::any()).will(invoke(E2EEndpointDestroyStub));
    // 硬件
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<u32>(0)), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    // channel 跳过平台层
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    // socket
    MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue((Hccl::Socket*)0xab));
    // port / device
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&devPort))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    // CCU（Init 中 TryInitCcuInstance 永不走，但以防万一）
    MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).stubs().will(returnValue(HCCL_SUCCESS));

    // ---- init CommMems + register tags ----
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank->Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    CommMems* commMems = myRank->GetCommMems();
    ASSERT_NE(commMems, nullptr);

    auto regMem = [&](const std::string& tag, void* addr) {
        CommMem m{COMM_MEM_TYPE_DEVICE, addr, 1024};
        void* h = nullptr;
        EXPECT_EQ(commMems->CommRegMem(tag, m, &h), HCCL_SUCCESS);
    };
    regMem("A", (void*)0xA000);
    regMem("B", (void*)0xB000);
    regMem("C", (void*)0xC000);

    // ---- create 2 endpoint × 2 channel ----
    EndpointDesc epA, epB;
    CreateEndpointDesc(epA, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    CreateEndpointDesc(epB, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "3.0.0.0");

    HcclChannelDesc channelDesc[2]{};
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = epA;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[0].memHandles = nullptr;
    channelDesc[0].memHandleNum = 0;

    channelDesc[1].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[1].remoteRank = 1;
    channelDesc[1].notifyNum = 2;
    channelDesc[1].localEndpoint = epB;
    channelDesc[1].remoteEndpoint = rmtEp;
    channelDesc[1].memHandles = nullptr;
    channelDesc[1].memHandleNum = 0;

    std::vector<HcommChannelDesc> hcommDesc(2);
    for (u32 i = 0; i < 2; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 2, "test", hcommDesc), HCCL_SUCCESS);

    // ---- Round 1: CreateChannels ----
    std::vector<ChannelHandle> hostChannelHandles1(2);
    std::vector<std::vector<MemHandle>> allHandles1(2);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AIV, channelDesc, 2, hcommDesc, hostChannelHandles1.data(), allHandles1),
        HCCL_SUCCESS);

    // cclBuffer + A + B + C = 4 tags, 2 endpoints → 8 HcommMemReg calls
    EXPECT_EQ(s_e2eRegCount, 8);
    EXPECT_EQ(s_e2eUnregCount, 0);

    // ---- Round 1.5: 无 CommMems 变更，版本不变跳过 ----
    std::vector<HcommChannelDesc> hcommDesc1b(2);
    for (u32 i = 0; i < 2; ++i) {
        hcommDesc1b[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 2, "test", hcommDesc1b), HCCL_SUCCESS);
    std::vector<ChannelHandle> hostChannelHandles1b(2);
    std::vector<std::vector<MemHandle>> allHandles1b(2);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AIV, channelDesc, 2, hcommDesc1b, hostChannelHandles1b.data(), allHandles1b),
        HCCL_SUCCESS);

    // 版本不变，RegisterMemory 跳过，无新增 HcommMemReg
    EXPECT_EQ(s_e2eRegCount, 8);

    // ---- UnregMemByTag("B") ----
    EXPECT_EQ(myRank->UnregMemByTag("B"), HCCL_SUCCESS);
    EXPECT_EQ(s_e2eUnregCount, 2); // epA + epB 各 1 次

    // ---- Round 2: 版本变更后重新注册 ----
    std::vector<HcommChannelDesc> hcommDesc2(2);
    for (u32 i = 0; i < 2; ++i) {
        hcommDesc2[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank->config_);
    }
    EXPECT_EQ(myRank->BatchCreateSockets(channelDesc, 2, "test", hcommDesc2), HCCL_SUCCESS);
    std::vector<ChannelHandle> hostChannelHandles2(2);
    std::vector<std::vector<MemHandle>> allHandles2(2);
    EXPECT_EQ(
        myRank->BatchCreateChannels(
            COMM_ENGINE_AIV, channelDesc, 2, hcommDesc2, hostChannelHandles2.data(), allHandles2),
        HCCL_SUCCESS);

    // 版本变更，"B"已不在 CommMems 中，"B"不会被注册
    // cclBuffer+A+C 已存在不触发 HcommMemReg，regCount 不变
    EXPECT_EQ(s_e2eRegCount, 8);

    // 在 mock 失效前主动析构 EndpointMgr，确保 TaggedMemMap 析构时
    // HcommMemUnreg / HcommEndpointDestroy 仍是 mock，不会调真函数操作假 handle
    myRank->endpointMgr_.reset();
}
