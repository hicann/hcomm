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
#include <iostream>
#include <unistd.h>

#define private public
#define protected public

#include "base_config.h"
#include "ccu_component.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator.h"
#include "communicator_impl.h"
#include "detour_service.h"
#include "env_config.h"
#include "json_parser.h"
#include "phy_topo.h"
#include "phy_topo_builder.h"
#include "rank_gph.h"
#include "rank_graph_builder.h"
#include "rank_table.h"
#include "ranktable_builder.h"
#include "sal.h"

#undef private
#undef protected

using namespace Hccl;

class RankGraphBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GenTopoFile();
        std::cout << "RankGraphBuilderTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        DelTopoFile();
        std::cout << "RankGraphBuilderTest TearDown" << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankGraphBuilderTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RankGraphBuilderTest TearDown" << std::endl;
    }
};

TEST_F(RankGraphBuilderTest, Ut_Build_When_Normal_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.Build(RankTable2p, "topo.json", 0);
    EXPECT_NE(nullptr, rankGraph);
    auto path1 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_NE(1, path1.size());
    auto path2 = rankGraph->GetPaths(0, 1, 0);
    EXPECT_NE(1, path2.size());
}

TEST_F(RankGraphBuilderTest, Ut_BuildRankGraph_When_Normal_Expect_Success)
{
    // when
    MOCKER_CPP(&RankGraphBuilder::BuildFromRankTable).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraph::InitInnerRanks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::UpdateTopoInstForMyRankOnly).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::SetEndpointDesc).stubs();
    // then
    RankGraphBuilder rankGraphBuilder;
    Hccl::RankTableInfo tmpRankTable;
    tmpRankTable.rankCount = 3;
    rankGraphBuilder.rankTable_ = make_unique<Hccl::RankTableInfo>(tmpRankTable);
    rankGraphBuilder.myRank_ = 0;
    EXPECT_NO_THROW(rankGraphBuilder.BuildRankGraph());
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_1pRankTable_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::string topoPath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/topo/new_topo_builder/rank_graph_builder/1ptopo.json"};
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.Build(RankTable1p, topoPath, 0);
    EXPECT_NE(nullptr, rankGraph);
    auto path1 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_EQ(0, path1.size());
    auto path2 = rankGraph->GetPaths(0, 1, 0);
    EXPECT_EQ(0, path2.size());
    vector<u32> subRankIds = {0};
    auto subRankGraph = rankGraph->CreateSubRankGraph(subRankIds);
    EXPECT_EQ(1, subRankGraph->GetInnerRankSize());
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_OnePTopoFileWithoutEdge_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::string topoPath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/topo/new_topo_builder/rank_graph_builder/1ptopo_without_edge.json"};
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.Build(RankTable1p, topoPath, 0);
    EXPECT_NE(nullptr, rankGraph);
    auto rankSize = rankGraph->GetRankSize();
    EXPECT_EQ(rankSize, 1);
    auto peer = rankGraph->GetPeer(rankGraph->GetMyRank());
    ASSERT_NE(peer, nullptr);
    EXPECT_EQ(peer->GetLocalId(), 0);
}

TEST_F(RankGraphBuilderTest, Ut_BuildFromRankTable_When_ValidRankTable_Expect_Success)
{
    // 校验BuildFromRankTable的Add(RankId, Peer)
    // when
    MOCKER_CPP(&PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    // 本用例只验证 RankTable 建图，不依赖物理拓扑构建。
    MOCKER_CPP(&RankGraphBuilder::AddTopoDescFabricInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraph::InitInnerRanks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::UpdateTopoInstForMyRankOnly).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::AddFabricInfo).stubs().will(ignoreReturnValue());
    // then
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph;
    EXPECT_NO_THROW(rankGraph = rankGraphBuilder.Build(RankTable3p, "topo.json", 0));
    EXPECT_NE(rankGraph, nullptr);
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_TopoWithoutNetLayer_Expect_UseRankTableLayers)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.Build(RankTable_4p, "topo.json", 0);
    ASSERT_NE(nullptr, rankGraph);
    const std::set<u32> expectedLevels = {0, 1, 2};
    EXPECT_EQ(3, rankGraph->GetLevelNum());
    EXPECT_EQ(expectedLevels, rankGraph->GetLevels(0));
    std::vector<std::string> netIds = {"az0-rack0", "az0", "all"};
    for (s32 rankId = 0; rankId < 4; rankId++) {
        for (u32 netLayer = 0; netLayer < 3; netLayer++) {
            const NetInstance *fabGroup = rankGraph->GetNetInstanceByRankId(netLayer, rankId);
            ASSERT_NE(nullptr, fabGroup);
            EXPECT_EQ(netIds[netLayer], fabGroup->GetNetInstId());

            EXPECT_EQ(true, fabGroup->HasNode(NetInstance::Peer(rankId, 0, 0, 0).GetLocalId()));
            if (netLayer == 0) {
                EXPECT_EQ(NetType::TOPO_FILE_DESC, fabGroup->GetNetType());
            } else if (netLayer == 1) {
                EXPECT_EQ(NetType::CLOS, fabGroup->GetNetType());
            } else if (netLayer == 2) {
                EXPECT_EQ(NetType::CLOS, fabGroup->GetNetType());
            }
        }
    }

    const NetInstance *layer0 = rankGraph->GetNetInstanceByNetInstId(0, "az0-rack0");
    ASSERT_NE(nullptr, layer0);
    EXPECT_TRUE(layer0->fabrics.empty());
    EXPECT_EQ(4, layer0->vGraph.nodes.size());

    std::vector<NetInstance::Path> pathsLayer0 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_EQ(1, pathsLayer0.size());
    // GetPaths检查边0 - 1的level1的边 peer0->net0、net0->peer1、peer0->net1、net1->peer1
    std::vector<NetInstance::Path> pathsLayer1 = rankGraph->GetPaths(1, 0, 1);
    std::vector<std::string> ips1 = {"IpAddress[AF=v4, addr=192.168.101.1]", "IpAddress[AF=v4, addr=124.112.1.1]"};
    std::vector<std::string> ips2 = {"IpAddress[AF=v4, addr=192.168.101.11]", "IpAddress[AF=v4, addr=124.112.1.4]"};
    EXPECT_EQ(1, pathsLayer1.size());
    EXPECT_EQ(2, pathsLayer1[0].links.size());

    EXPECT_EQ(2, pathsLayer1[0].links[0].GetHop());
    EXPECT_EQ(LinkType::PEER2NET, pathsLayer1[0].links[0].GetType());
    EXPECT_EQ(std::set<Hccl::LinkProtocol>{LinkProtocol::UB_CTP}, pathsLayer1[0].links[0].GetLinkProtocols());
    EXPECT_EQ(0, pathsLayer1[0].links[0].GetSourceNode()->GetNodeId());
    EXPECT_EQ(4294967296, pathsLayer1[0].links[0].GetTargetNode()->GetNodeId());
    EXPECT_EQ(htonl(0xC0A86501), pathsLayer1[0].links[0].GetSourceIface()->GetAddr().GetBinaryAddress().addr.s_addr);

    EXPECT_EQ(2, pathsLayer1[0].links[1].GetHop());
    EXPECT_EQ(LinkType::PEER2NET, pathsLayer1[0].links[1].GetType());
    EXPECT_EQ(std::set<Hccl::LinkProtocol>{LinkProtocol::UB_CTP}, pathsLayer1[0].links[0].GetLinkProtocols());
    EXPECT_EQ(4294967296, pathsLayer1[0].links[1].GetSourceNode()->GetNodeId());
    EXPECT_EQ(1, pathsLayer1[0].links[1].GetTargetNode()->GetNodeId());
    EXPECT_EQ(htonl(0xC0A8650B), pathsLayer1[0].links[1].GetTargetIface()->GetAddr().GetBinaryAddress().addr.s_addr);

    std::vector<NetInstance::Path> pathsLayer2 = rankGraph->GetPaths(2, 2, 3);
    EXPECT_EQ(1, pathsLayer2.size());
    EXPECT_EQ(2, pathsLayer2[0].links.size());
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_SameTopoInstHasDifferentPlanes_Expect_CreateFabricByPlaneId)
{
    PhyTopo::GetInstance()->Clear();
    const std::string rankTableString = R"({
        "version": "2.0",
        "rank_count": 1,
        "rank_list": [
            {
                "rank_id": 0,
                "device_id": 0,
                "local_id": 0,
                "level_list": [
                    {
                        "net_layer": 0,
                        "net_instance_id": "rank-group-0",
                        "net_type": "TOPO_FILE_DESC",
                        "rank_addr_list": [
                            {
                                "addr_type": "IPV4",
                                "addr": "192.168.0.1",
                                "ports": ["0/1"],
                                "plane_id": "planeA"
                            },
                            {
                                "addr_type": "IPV4",
                                "addr": "192.168.0.2",
                                "ports": ["0/2"],
                                "plane_id": "planeB"
                            }
                        ]
                    }
                ]
            }
        ]
    })";
    const std::string topoString = R"({
        "version": "2.0",
        "peer_count": 1,
        "peer_list": [{"local_id": 0}],
        "edge_count": 2,
        "edge_list": [
            {
                "link_type": "PEER2NET",
                "protocols": ["UB_CTP"],
                "topo_type": "CLOS",
                "topo_instance_id": 7,
                "local_a": 0,
                "local_a_ports": ["0/1"],
                "position": "DEVICE"
            },
            {
                "link_type": "PEER2NET",
                "protocols": ["UB_CTP"],
                "topo_type": "CLOS",
                "topo_instance_id": 7,
                "local_a": 0,
                "local_a_ports": ["0/2"],
                "position": "DEVICE"
            }
        ]
    })";

    JsonParser parser;
    RankTableInfo rankTableInfo;
    TopoInfo topoInfo;
    parser.ParseString(rankTableString, rankTableInfo);
    parser.ParseString(topoString, topoInfo);

    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0);
    ASSERT_NE(nullptr, rankGraph);
    NetInstance *layer0 = rankGraph->GetNetInstanceByNetInstId(0, "rank-group-0");
    ASSERT_NE(nullptr, layer0);
    ASSERT_EQ(2, layer0->GetFabrics().size());
    std::set<PlaneId> planeIds;
    for (const auto &fabric : layer0->GetFabrics()) {
        ASSERT_NE(nullptr, fabric);
        planeIds.insert(fabric->GetPlaneId());
    }
    EXPECT_EQ((std::set<PlaneId>{"planeA", "planeB"}), planeIds);
}

TEST_F(RankGraphBuilderTest, Init_error)
{
    CommunicatorImpl comm;
    comm.initFlag = true;
    CommParams commParams;
    std::unique_ptr<RankGraph> virtualTopo = std::make_unique<RankGraph>(0);
    HcclCommConfig subConfig;
    DevId inputDevLogicId = 0;
    auto ret = comm.Init(commParams, virtualTopo, subConfig, inputDevLogicId);
    EXPECT_EQ(HCCL_E_INTERNAL, ret);
}

TEST_F(RankGraphBuilderTest, CreateSubFabGroups_error)
{
    CommunicatorImpl comm;
    comm.initFlag = false;
    CommParams params;
    CommunicatorImpl subCommImpl;
    std::vector<u32> rankIds = {0, 1};
    auto ret = comm.CreateSubComm(params, rankIds, &subCommImpl);
    EXPECT_EQ(HCCL_E_INTERNAL, ret);
}

TEST_F(RankGraphBuilderTest, CreateSubFabGroups_error_1)
{
    CommunicatorImpl comm;
    comm.initFlag = false;
    CommParams params;
    CommunicatorImpl subCommImpl;
    HcclCommConfig subConfig;
    std::vector<u32> rankIds = {0, 1};
    auto ret = comm.CreateSubComm(params, rankIds, &subCommImpl, subConfig);
    EXPECT_EQ(HCCL_E_INTERNAL, ret);
}

TEST_F(RankGraphBuilderTest, Ut_ConstructConnI_When_PortMapEmpty_Expect_OnlyPcieCreatesD2hIface)
{
    const std::map<std::string, std::vector<IpAddress>> emptyPortAddrMap;
    auto pcieIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"0/0"}, AddrPosition::DEVICE, LinkType::PEER2PEER,
        std::set<LinkProtocol>{LinkProtocol::PCIE});
    MOCKER(HrtRaSocketGetVnicIpInfos).stubs();

    auto pcieNetIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
        pcieIface, emptyPortAddrMap, TopoType::CLOS, 0, 0);

    ASSERT_EQ(1, pcieNetIfaces.size());
    EXPECT_EQ(std::set<std::string>{"d2h"}, pcieNetIfaces[0]->GetPorts());
    EXPECT_EQ(std::set<LinkProtocol>{LinkProtocol::PCIE}, pcieNetIfaces[0]->GetLinkProtocols());

    auto unmatchedIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"0/9"}, AddrPosition::DEVICE, LinkType::PEER2PEER,
        std::set<LinkProtocol>{LinkProtocol::UB_CTP});
    auto unmatchedNetIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
        unmatchedIface, emptyPortAddrMap, TopoType::CLOS, 0, 0);

    EXPECT_TRUE(unmatchedNetIfaces.empty());
}
TEST_F(RankGraphBuilderTest, Ut_RankGraphBuilderRecoverBuild_When_Invalid_Expect_InvalidParamsException)
{
    RankTableInfo rankTableInfo;
    TopoInfo topoInfo;
    RankGraphBuilder rankGraphBuilder;
    EXPECT_THROW(rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0), InvalidParamsException);
}

TEST_F(RankGraphBuilderTest, Ut_RankGraphBuilderBuild_When_Empty_Expect_InvalidParamsException)
{
    PhyTopo::GetInstance()->initFlag = false;
    std::string rankTable;
    RankGraphBuilder rankGraphBuilder;
    std::string topoPath;
    EXPECT_THROW(rankGraphBuilder.Build(rankTable, topoPath, 0), InvalidParamsException);
    PhyTopo::GetInstance()->initFlag = true;
}
