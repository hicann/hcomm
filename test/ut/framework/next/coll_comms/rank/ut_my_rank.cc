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
#define private public
#include "my_rank.h"
#undef private
#include "hccl_comm_pub.h"
#include "llt_hccl_stub_rank_graph.h"

using namespace hccl;

class MyRankTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MyRankTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MyRankTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in MyRankTest SetUP" << std::endl;
        rankIpPortMap = std::make_shared<std::unordered_map<u32, std::unordered_map<Hccl::IpAddress, u32>>>();
        (*rankIpPortMap)[0][Hccl::IpAddress("1.0.0.0")] = 16666;
        (*rankIpPortMap)[1][Hccl::IpAddress("2.0.0.0")] = 16666;
        (*rankIpPortMap)[2][Hccl::IpAddress("0.0.0.0")] = 16666;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in MyRankTest TearDown" << std::endl;
    }

    void CreateCclBuffer(HcclMem& cclBuffer) {
        cclBuffer.addr = (void*)0xab;
        cclBuffer.size = 1024;
        cclBuffer.type = HCCL_MEM_TYPE_DEVICE;
    }

    void CreateEndpointDesc(EndpointDesc& ep, CommProtocol protocol, const std::string& ip) {
        ep.protocol = protocol;
        ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ep.commAddr.addr = Hccl::IpAddress(ip).GetBinaryAddress().addr;
        ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    }

    void MockerFuncs()
    {
        MOCKER_CPP(&Hccl::SocketManager::GetConnectedSocket).stubs().with(mockcpp::any()).will(returnValue((Hccl::Socket*)0xab));
        MOCKER_CPP(&hccl::CommMems::GetTagMemoryHandles).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&hcomm::EndpointMgr::RegisterMemory).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(HcommCcuInsCreate).stubs().with(mockcpp::any()).will(returnValue(CcuResult::CCU_SUCCESS));
        MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).stubs().will(returnValue(HCCL_SUCCESS));
    }

    uint32_t DEFAULT_MODE = 0;
    uint32_t AICPU_TS_MODE = 2;
    uint32_t CCU_MS_MODE = 5;
    uint32_t CCU_SCHED_MODE = 6;
    Hccl::RankIpPortMapPtr rankIpPortMap;
};

TEST_F(MyRankTest, Ut_When_QueryListenPort_Listen_Port_Expect_SUCCESS)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().with(mockcpp::any(), outBoundP(&devPort)).will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "2.0.0.0");

    uint32_t listenPort;
    HcommChannelDesc desc;
    HcclResult ret = myRank.QueryListenPort(0, 1, localEp, rmtEp, listenPort, desc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listenPort, devPort);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_SERVER);

    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_ROCE, "0.0.0.0");
    ret = myRank.QueryListenPort(0, 2, localEp, rmtEp2, listenPort, desc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listenPort, devPort);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_CLIENT);
}

TEST_F(MyRankTest, Ut_When_QueryListenPort_InValid_Port_Expect_E_PARA)
{
    uint32_t devPort = 1919000;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().with(mockcpp::any(), outBoundP(&devPort)).will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_ROCE, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_ROCE, "2.0.0.0");

    uint32_t listenPort;
    HcommChannelDesc desc;
    HcclResult ret = myRank.QueryListenPort(0, 1, localEp, rmtEp, listenPort, desc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(MyRankTest, Ut_When_BatchCreateChannels_Expect_SUCCESS)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().with(mockcpp::any(), outBoundP(&devPort)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId).stubs().with(mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();
    ChannelHandle channelHandle = 0xab;
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank.Init(cclBuffer, 2, 2), HCCL_SUCCESS);
    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UB_MEM, "0.0.0.0");

    HcclChannelDesc channelDesc[3];
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;
    channelDesc[1].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[1].remoteRank = 1;
    channelDesc[1].notifyNum = 2;
    channelDesc[1].localEndpoint = localEp;
    channelDesc[1].remoteEndpoint = rmtEp;
    channelDesc[2].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[2].remoteRank = 2;
    channelDesc[2].notifyNum = 2;
    channelDesc[2].localEndpoint = localEp;
    channelDesc[2].remoteEndpoint = rmtEp2;

    u32 channelIdx0 = 0u;
    u32 channelIdx1 = 1u;
    u32 channelIdx2 = 2u;
    u32 RmtEp1reuseIdx0 = 0u;
    u32 RmtEp1reuseIdx1 = 1u;
    u32 RmtEp2reuseIdx0 = 0u;

    std::vector<HcommChannelDesc> hcommDesc(3);
    for (u32 i = 0; i < 3; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank.config_);
    }
    EXPECT_EQ(myRank.BatchCreateSockets(channelDesc, 1, "test", hcommDesc), HCCL_SUCCESS);
    std::vector<ChannelHandle> hostChannelHandles(3);
    ChannelHandle *hostChannelHandleList = hostChannelHandles.data();
    EXPECT_EQ(myRank.BatchCreateChannels(COMM_ENGINE_AICPU_TS, channelDesc, 1, hcommDesc, hostChannelHandleList), HCCL_SUCCESS);
    EXPECT_EQ(myRank.newChannels_.size(), 1);
    EXPECT_EQ(myRank.newChannels_[0], std::make_pair(channelIdx0, RmtEp1reuseIdx0));

    EXPECT_EQ(myRank.BatchCreateSockets(channelDesc, 2, "test", hcommDesc), HCCL_SUCCESS);
    EXPECT_EQ(myRank.BatchCreateChannels(COMM_ENGINE_AICPU_TS, channelDesc, 2, hcommDesc, hostChannelHandleList), HCCL_SUCCESS);
    EXPECT_EQ(myRank.newChannels_.size(), 1);
    EXPECT_EQ(myRank.newChannels_[0], std::make_pair(channelIdx1, RmtEp1reuseIdx1));

    EXPECT_EQ(myRank.BatchCreateSockets(channelDesc, 3, "test", hcommDesc), HCCL_SUCCESS);
    EXPECT_EQ(myRank.BatchCreateChannels(COMM_ENGINE_AICPU_TS, channelDesc, 3, hcommDesc, hostChannelHandleList), HCCL_SUCCESS);
    EXPECT_EQ(myRank.newChannels_.size(), 1);
    EXPECT_EQ(myRank.newChannels_[0], std::make_pair(channelIdx2, RmtEp2reuseIdx0));

    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_AGAIN));
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().with(mockcpp::any()).will(returnValue((s32)(1)));
    EXPECT_EQ(myRank.BatchConnectChannels(channelDesc, hostChannelHandleList, 3), HCCL_E_TIMEOUT);
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGetStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_TIMEOUT));
    EXPECT_EQ(myRank.BatchConnectChannels(channelDesc, hostChannelHandleList, 3), HCCL_E_TIMEOUT);
    unsetenv("HCCL_DFS_CONFIG");
}

// 测试Init时用户未配置展开模式时，读取环境变量配置
TEST_F(MyRankTest, Ut_Init_When_Default_Mode_Expect_Set_By_Env)
{
    setenv("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", 1);
    MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).stubs().will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t defaultOpExpansionMode = DEFAULT_MODE;
    EXPECT_EQ(myRank.Init(cclBuffer, defaultOpExpansionMode, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank.opExpansionMode_, CCU_SCHED_MODE);
    unsetenv("HCCL_OP_EXPANSION_MODE");
}

// 测试Init时ccu驱动拉起失败回退到aicpu
TEST_F(MyRankTest, Ut_Init_When_Ccu_Driver_Fail_Expect_Fallback_Aicpu)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(HcommCcuInsCreate).stubs().will(returnValue(CcuResult::CCU_E_DRV_BUSY));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank.Init(cclBuffer, opExpansionModeMs, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank.opExpansionMode_, AICPU_TS_MODE);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试Init时ccu ms资源不足回退到sched
TEST_F(MyRankTest, Ut_Init_When_Ccu_Ms_Insufficient_Expect_Fallback_Sched)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(HcommCcuInsCreate).stubs()
        .will(returnValue(CcuResult::CCU_E_UNAVAIL))
        .then(returnValue(CcuResult::CCU_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank.Init(cclBuffer, opExpansionModeMs, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank.opExpansionMode_, CCU_SCHED_MODE);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试Init时ccu ms和sched资源不足回退到aicpu
TEST_F(MyRankTest, Ut_Init_When_Ccu_Ms_And_Sched_Insufficient_Expect_Fallback_Aicpu)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(HcommCcuInsCreate).stubs().will(returnValue(CcuResult::CCU_E_UNAVAIL));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank.Init(cclBuffer, opExpansionModeMs, 2), HCCL_SUCCESS);
    EXPECT_EQ(myRank.opExpansionMode_, AICPU_TS_MODE);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试Init在申请资源时出现其他报错时失败
TEST_F(MyRankTest, Ut_Init_When_Resource_Fail_Expect_Fail)
{
    setenv("HCCL_CCU_CUSTOM_OP_MODE", "1", 1);
    MOCKER(HcommCcuInsCreate).stubs().will(returnValue(CcuResult::CCU_E_PARA));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);

    uint32_t opExpansionModeMs = CCU_MS_MODE;
    EXPECT_EQ(myRank.Init(cclBuffer, opExpansionModeMs, 2), HCCL_E_PARA);
    unsetenv("HCCL_CCU_CUSTOM_OP_MODE");
}

// 测试BatchCreateChannels在资源不足时销毁新申请的channel
TEST_F(MyRankTest, St_BatchCreateChannels_When_Resource_fallback_Expect_Return_HCCL_E_UNAVAIL)
{
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().with(mockcpp::any(), outBoundP(&devPort)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommEndpointStartListen).stubs().with(mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(HcommChannelDestroy).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank.Init(cclBuffer, 5, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UBC_CTP, "1.0.0.0");
    EndpointDesc rmtEp1;
    CreateEndpointDesc(rmtEp1, COMM_PROTOCOL_UBC_CTP, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UBC_CTP, "3.0.0.0");

    HcclChannelDesc channelDesc[5];
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp1;
    }
    for (int i = 2; i < 5; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
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
    ChannelHandle *hostChannelHandleList = hostChannelHandles.data();
    EXPECT_EQ(myRank.BatchCreateChannels(COMM_ENGINE_CCU, channelDesc, 5, hcommDesc, hostChannelHandleList), HCCL_E_UNAVAIL);
    EXPECT_EQ(myRank.newChannels_.size(), 0);

    // 获取到rmtEp1的endpointPair
    RankIdPair rankIdPair1 = std::make_pair(0, 1);
    EndpointDescPair endpointDescPair1 = std::make_pair(localEp, rmtEp1);
    RankPair* rankPair1 = nullptr;
    hcomm::EndpointPair* endpointPair1 = nullptr;
    myRank.rankPairMgr_->Get(rankIdPair1, rankPair1);
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
    myRank.rankPairMgr_->Get(rankIdPair2, rankPair2);
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
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().with(mockcpp::any(), outBoundP(&devPort)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommEndpointStartListen).stubs().with(mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(HcommChannelDestroy).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    
    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank.Init(cclBuffer, 5, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UBC_CTP, "1.0.0.0");
    EndpointDesc rmtEp1;
    CreateEndpointDesc(rmtEp1, COMM_PROTOCOL_UBC_CTP, "2.0.0.0");
    EndpointDesc rmtEp2;
    CreateEndpointDesc(rmtEp2, COMM_PROTOCOL_UBC_CTP, "3.0.0.0");

    HcclChannelDesc channelDesc[5];
    for (int i = 0; i < 2; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
        channelDesc[i].remoteRank = 1;
        channelDesc[i].notifyNum = 2;
        channelDesc[i].localEndpoint = localEp;
        channelDesc[i].remoteEndpoint = rmtEp1;
    }
    for (int i = 2; i < 5; i++) {
        channelDesc[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
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
        .will(returnValue(static_cast<int>(HCCL_SUCCESS))) // 第一次调用，到rmtEp1的channel1成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS))) // 第一次调用，到rmtEp1的channel2成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS))) // 第一次调用，到rmtEp2的channel1成功
        .then(returnValue(static_cast<int>(HCCL_SUCCESS))) // 第二次调用，到rmtEp2的channel2成功
        .then(returnValue(static_cast<int>(HCCL_E_UNAVAIL))); // 第二次调用，到rmtEp2的channel3失败
    std::vector<HcommChannelDesc> hcommDesc(5);
    std::vector<ChannelHandle> hostChannelHandles(5);
    ChannelHandle *hostChannelHandleList = hostChannelHandles.data();
    // 第一次调用BatchCreateChannels成功，创建3个channel
    EXPECT_EQ(myRank.BatchCreateChannels(COMM_ENGINE_CCU, channelDesc, 3, hcommDesc, hostChannelHandleList), HCCL_SUCCESS);
    EXPECT_EQ(myRank.newChannels_.size(), 3);
    u32 channelIdx0 = 0u;
    u32 channelIdx1 = 1u;
    u32 channelIdx2 = 2u;
    u32 RmtEp1reuseIdx0 = 0u;
    u32 RmtEp1reuseIdx1 = 1u;
    u32 RmtEp2reuseIdx0 = 0u;
    EXPECT_EQ(myRank.newChannels_[0], std::make_pair(channelIdx0, RmtEp1reuseIdx0));
    EXPECT_EQ(myRank.newChannels_[1], std::make_pair(channelIdx1, RmtEp1reuseIdx1));
    EXPECT_EQ(myRank.newChannels_[2], std::make_pair(channelIdx2, RmtEp2reuseIdx0));

    // 获取到rmtEp1的endpointPair
    RankIdPair rankIdPair1 = std::make_pair(0, 1);
    EndpointDescPair endpointDescPair1 = std::make_pair(localEp, rmtEp1);
    RankPair* rankPair1 = nullptr;
    hcomm::EndpointPair* endpointPair1 = nullptr;
    myRank.rankPairMgr_->Get(rankIdPair1, rankPair1);
    rankPair1->GetEndpointPair(endpointDescPair1, endpointPair1);

    // 获取到rmtEp2的endpointPair
    RankIdPair rankIdPair2 = std::make_pair(0, 2);
    EndpointDescPair endpointDescPair2 = std::make_pair(localEp, rmtEp2);
    RankPair* rankPair2 = nullptr;
    hcomm::EndpointPair* endpointPair2 = nullptr;
    myRank.rankPairMgr_->Get(rankIdPair2, rankPair2);
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
    EXPECT_EQ(myRank.BatchCreateChannels(COMM_ENGINE_CCU, channelDesc, 5, hcommDesc, hostChannelHandleList), HCCL_E_UNAVAIL);
    EXPECT_EQ(myRank.newChannels_.size(), 0);

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
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    ChannelHandle channel = 0x12345;
    void *buffer = nullptr;
    uint64_t size = 0;
    HcclResult ret = myRank.ChannelGetHcclBuffer(channel, &buffer, &size);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_Normal_Expect_SUCCESS)
{
    MOCKER(hcomm::ChannelProcess::ChannelGetRemoteMems)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    char** memInfo = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank.ChannelGetRemoteMems(channel, &memNum, &remoteMem, &memInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_RemoteMemNull_Expect_E_PTR)
{
    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    ChannelHandle channel = 0x12345;
    char** memInfo = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank.ChannelGetRemoteMems(channel, &memNum, nullptr, &memInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_MemInfoNull_Expect_E_PTR)
{
    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = myRank.ChannelGetRemoteMems(channel, &memNum, &remoteMem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_When_ChannelGetRemoteMems_MemNumNull_Expect_E_PTR)
{
    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    ChannelHandle channel = 0x12345;
    CommMem* remoteMem = nullptr;
    char** memInfo = nullptr;
    HcclResult ret = myRank.ChannelGetRemoteMems(channel, nullptr, &remoteMem, &memInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(MyRankTest, Ut_ChannelDescHccl2Hcomm_When_UbcCtp_Sets_CommDomainQos_FromCommConfig)
{
    CommConfig commConfig("ut");
    ASSERT_EQ(commConfig.SetConfigHcclQos(5u), HCCL_SUCCESS);
    HcclChannelDesc in{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_CTP;
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

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CCU_MSModel_WithCommConfig)
{
    HcommChannelDesc in{};
    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    myRank.opExpansionMode_ = CCU_MS_MODE;
    HcclResult ret = myRank.ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, in);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(in.ubAttr.sqDepth, 128);
}

TEST_F(MyRankTest, Ut_ConfigSqDepthByExpansionMode_When_CCU_SCHEDModel_WithCommConfig)
{
    HcommChannelDesc in{};
    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);
    myRank.opExpansionMode_ = CCU_SCHED_MODE;
    HcclResult out = myRank.ConfigSqDepthByExpansionMode(COMM_ENGINE_CCU, in);
    EXPECT_EQ(out, HCCL_SUCCESS);
    EXPECT_EQ(in.ubAttr.sqDepth, 16);
}

// LLT测试：验证socketTag格式正确构建（commTag_engine_X格式）
// 测试流程：MyRank::BatchCreateSockets中socketTag = commTag + "_engine_" + std::to_string(engine)
//         -> EndpointPair::GetSocket -> SocketConfig解析commTag
// 使用UB_MEM协议和DEVICE endpoint，验证socketTag格式是否正确
TEST_F(MyRankTest, LLT_SocketTagFormat_CommTagEngine_Success)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().with(any(), outBoundP(&devPort)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId).stubs().with(any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommCollectiveChannelCreate)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(HcommChannelUpdateMemInfo)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(hcomm::ChannelProcess::ChannelGetStatus)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank.Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    // 设置DEVICE侧endpoint，使用UB_MEM协议（与通过测试一致）
    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    HcclChannelDesc channelDesc[1];
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;

    std::vector<HcommChannelDesc> hcommDesc(1);
    for (u32 i = 0; i < 1; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank.config_);
    }

    std::vector<ChannelHandle> hostChannelHandles(1);
    ChannelHandle *hostChannelHandleList = hostChannelHandles.data();

    // 验证: socketTag = "testCommTag_engine_5" (engine=5对应COMM_ENGINE_AIV)
    // BatchCreateSockets内部会将socketTag传递给SocketConfig进行解析
    EXPECT_EQ(myRank.BatchCreateSockets(channelDesc, 1, "testCommTag_engine_5", hcommDesc), HCCL_SUCCESS);
    EXPECT_EQ(myRank.BatchCreateChannels(COMM_ENGINE_AIV, channelDesc, 1, hcommDesc, hostChannelHandleList), HCCL_SUCCESS);
    EXPECT_EQ(myRank.newChannels_.size(), 1);

    unsetenv("HCCL_DFS_CONFIG");
}

// LLT测试：验证带下划线的commTag能正确构建socketTag并解析
// 测试流程同上，使用UB_MEM协议和DEVICE endpoint
TEST_F(MyRankTest, LLT_SocketTagFormat_CommTagWithUnderscore_Success)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().with(any(), outBoundP(&devPort)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId).stubs().with(any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommCollectiveChannelCreate)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(HcommChannelUpdateMemInfo)
        .stubs()
        .will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(hcomm::ChannelProcess::ChannelGetStatus)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    EXPECT_EQ(myRank.Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    // 设置DEVICE侧endpoint，使用UB_MEM协议
    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    HcclChannelDesc channelDesc[1];
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;

    std::vector<HcommChannelDesc> hcommDesc(1);
    for (u32 i = 0; i < 1; ++i) {
        hcommDesc[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDesc[i], myRank.config_);
    }

    std::vector<ChannelHandle> hostChannelHandles(1);
    ChannelHandle *hostChannelHandleList = hostChannelHandles.data();

    // 验证: socketTag = "my_comm_tag_engine_2" 格式（带下划线的commTag）
    EXPECT_EQ(myRank.BatchCreateSockets(channelDesc, 1, "my_comm_tag_engine_2", hcommDesc), HCCL_SUCCESS);
    EXPECT_EQ(myRank.BatchCreateChannels(COMM_ENGINE_AICPU_TS, channelDesc, 1, hcommDesc, hostChannelHandleList), HCCL_SUCCESS);
    EXPECT_EQ(myRank.newChannels_.size(), 1);

    unsetenv("HCCL_DFS_CONFIG");
}

TEST_F(MyRankTest, Ut_CreateChannels_When_CpuEngine_Success_Expect_HCCL_SUCCESS)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    uint32_t devPort = 60001;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().with(any(), outBoundP(&devPort)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::IRankGraph::GetDeviceId).stubs().with(any()).will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MockerFuncs();
    MOCKER(hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommCollectiveChannelCreate).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));
    MOCKER(hcomm::ChannelProcess::ChannelGetStatus).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclCommDfx::AddChannelRemoteRankId).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&ExchangeInfoMgr::BatchExchangeAndCheckConsistency).stubs().will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    HcclMem cclBuffer;
    CreateCclBuffer(cclBuffer);
    ASSERT_EQ(myRank.Init(cclBuffer, 2, 2), HCCL_SUCCESS);

    EndpointDesc localEp;
    CreateEndpointDesc(localEp, COMM_PROTOCOL_UB_MEM, "1.0.0.0");
    EndpointDesc rmtEp;
    CreateEndpointDesc(rmtEp, COMM_PROTOCOL_UB_MEM, "2.0.0.0");

    HcclChannelDesc channelDesc[1];
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UB_MEM;
    channelDesc[0].remoteRank = 1;
    channelDesc[0].notifyNum = 2;
    channelDesc[0].localEndpoint = localEp;
    channelDesc[0].remoteEndpoint = rmtEp;

    ChannelHandle channelHandles[1] = {0};
    ASSERT_EQ(myRank.CreateChannels(COMM_ENGINE_CPU, "testComm", channelDesc, 1, channelHandles), HCCL_SUCCESS);

    unsetenv("HCCL_DFS_CONFIG");
}
