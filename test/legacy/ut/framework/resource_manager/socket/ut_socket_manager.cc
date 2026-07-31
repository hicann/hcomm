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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include "socket_manager.h"
#include "socket_handle_manager.h"
#include "communicator_impl.h"
#include "ranktable_stub_clos.h"
#include "preempt_port_manager.h"
#include "host_socket_handle_manager.h"
#include "phy_topo.h"
#undef protected
#undef private

using namespace Hccl;

class SocketManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SocketManagerTest SetUP" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SocketManagerTest TearDown" << std::endl;
    }

    virtual void SetUp() {
        hccpSocketHandle = new int(0);
        MOCKER_CPP(&SocketHandleManager::Create)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hccpSocketHandle));
        MOCKER_CPP(&SocketHandleManager::Get)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hccpSocketHandle));
        MOCKER_CPP(&PreemptPortManager::ListenPreempt)
            .stubs()
            .will(ignoreReturnValue());
        SetLinks();

        std::cout << "A Test case in SocketManagerTest SetUP" << std::endl;
    }

    virtual void TearDown() {
        GlobalMockObject::verify();
        delete hccpSocketHandle;
        std::cout << "A Test case in SocketManagerTest TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress(RankId rankId=1)
    {
        IpAddress ipAddress(StringFormat("%u.0.0.0", rankId));
        return ipAddress;
    }

    void SetLinks()
    {
        links.clear();
        for (u32 i = 0; i < 4; i++) {
            PortDeploymentType portDeploymentType = PortDeploymentType::DEV_NET;
            LinkProtocol linkProtocol = LinkProtocol::UB_CTP;
            RankId localRankId = 0;
            RankId remoteRankId = 3 - i;
            
            LinkData tmpLink(portDeploymentType, linkProtocol, localRankId, remoteRankId,
                GetAnIpAddress(localRankId), GetAnIpAddress(remoteRankId));
            links.push_back(tmpLink);
        }
    }

    void *hccpSocketHandle;
    IpAddress           localIp;
    IpAddress           remoteIp;
    vector<LinkData>    links;
    CommunicatorImpl impl;
    u32 localRank = 0;
    u32 devicePhyId = 0;
    u32 listenPort = 60001;
};


TEST_F(SocketManagerTest, batch_create_sockets_should_ok) {
    // Given
    MOCKER_CPP(&SocketManager::BatchAddWhiteList).stubs();
    
    // when

    // then
    SocketManager socketMgr(impl, localRank, devicePhyId, listenPort);
    socketMgr.BatchCreateSockets(links);
    auto &serverSocketMap = SocketManager::GetServerSocketMap();
    for (const auto& sock: serverSocketMap) {
        EXPECT_EQ(sock.second->socketStatus, SocketStatus::LISTENING);
    }

    for (const auto& sock: socketMgr.connectedSocketMap) {
        if (sock.first.role == SocketRole::CLIENT) {
            EXPECT_EQ(sock.second->socketStatus, SocketStatus::CONNECT_STARTING);
        }
        std::cout << sock.first.remoteRank << " " << sock.second->socketStatus << std::endl;
    }
}


TEST_F(SocketManagerTest, batch_server_listen_async_conect_sockets_should_ok) {
    MOCKER_CPP(&SocketManager::BatchAddWhiteList).stubs();
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
 	auto link = links[0];
    Hccl::SocketConfig socketConfig(link.GetRemoteRankId(), link, "test");
    socketMgr.ServerListen(socketConfig);
    auto &serverSocketMap = SocketManager::GetServerSocketMap();
    for (const auto& sock: serverSocketMap) {
        EXPECT_EQ(sock.second->socketStatus, SocketStatus::LISTENING);
    }
    socketMgr.ConnectSockets(socketConfig);
    for (const auto& sock: socketMgr.connectedSocketMap) {
        if (sock.first.role == SocketRole::CLIENT) {
            EXPECT_EQ(sock.second->socketStatus, SocketStatus::CONNECT_STARTING);
        }
        std::cout << sock.first.remoteRank << " " << sock.second->socketStatus << std::endl;
    }
}

TEST_F(SocketManagerTest, test_ServerDeInit_and_GetServerListenSocket) {
    MOCKER_CPP(&SocketManager::BatchAddWhiteList).stubs();
    SocketManager socketMgr(impl, localRank, devicePhyId, listenPort);
    socketMgr.BatchCreateSockets(links);
    for(auto& link : links){
        auto portData = link.GetLocalPort();
        socketMgr.ServerDeInit(portData);
        socketMgr.GetServerListenSocket(portData);
    }
}

TEST_F(SocketManagerTest, Ut_ServerInitAll_Skip_Init_When_Env_not_Config) {
    EnvHostNicConfig envConfig;
    EnvHostNicConfig &fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclDeviceSocketPortRange = CfgField<std::vector<SocketPortRange>>{"HCCL_NPU_SOCKET_PORT_RANGE", {}, 
        [] (const std::string &s) -> std::vector<SocketPortRange> { return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE"); }};
    fakeEnvConfig.hcclDeviceSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));
    NewRankInfo rankInfo;
    EXPECT_NO_THROW(SocketManager::ServerInitAll(rankInfo));
}

TEST_F(SocketManagerTest, Ut_ServerInitAll_Skip_Init_When_Env_Config) {
    EnvHostNicConfig envConfig;
    EnvHostNicConfig &fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclDeviceSocketPortRange = CfgField<std::vector<SocketPortRange>>{"HCCL_NPU_SOCKET_PORT_RANGE", {{16666, 18888}}, 
        [] (const std::string &s) -> std::vector<SocketPortRange> { return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE"); }};
    fakeEnvConfig.hcclDeviceSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo2pclos.json"};
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath)
        .stubs()
        .will(returnValue(topoFilePath));
    MOCKER(HrtGetDevice)
        .stubs()
        .will(returnValue(0));
    RankGraphBuilder rankGraphBuilder;
    unique_ptr<RankGraph> graph = rankGraphBuilder.Build(RankTable2pClos, topoFilePath, 0);
    EXPECT_NE(nullptr, graph);
    NewRankInfo rankInfo = rankGraphBuilder.GetRankTableInfo()->ranks[0];
    EXPECT_NO_THROW(SocketManager::ServerInitAll(rankInfo));
}

TEST_F(SocketManagerTest, test_BatchCreateSockets_with_SocketConfig) {
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    auto link = links[0];
    Hccl::SocketConfig socketConfig(link.GetRemoteRankId(), link, "test");
    socketMgr.BatchCreateSockets(socketConfig);
    socketMgr.GetConnectedSocket(socketConfig);
}

TEST_F(SocketManagerTest, test_CheckServerPortListening_When_Port_Inconsistent_Expected_False) {
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    auto link = links[0];
    auto &serverSocketMap = SocketManager::GetServerSocketMap();
    serverSocketMap[link.GetLocalPort()] = std::make_shared<Socket>(hccpSocketHandle, link.GetLocalPort().GetAddr(), 60001, link.GetRemotePort().GetAddr(), "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    bool isListen = socketMgr.CheckServerPortListening(link.GetLocalPort(), 60002);
    EXPECT_FALSE(isListen);
}