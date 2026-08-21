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

#define private public
#define protected public

#include "rank_graph_builder.h"
#include "rank_gph.h"
#include "phy_topo_builder.h"
#include "phy_topo.h"
#include "detour_service.h"
#include "graph.h"
#include "rank_graph_test_data_builder.h"

#undef private
#undef protected

using namespace Hccl;
using namespace std;

class RankGraph64Plus1Test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "RankGraph64Plus1Test SetUP" << endl; }

    static void TearDownTestCase() { cout << "RankGraph64Plus1Test TearDown" << endl; }

    virtual void SetUp()
    {
        PhyTopo::GetInstance()->Clear();                       // PhyTopo是单例，每个用例开始前需要重置
        MOCKER_CPP(&DetourService::InsertDetourLinks).stubs(); // 64+1场景暂时不涉及绕路，将绕路接口打桩成空函数
        cout << "A Test case in RankGraph64Plus1Test SetUP" << endl;
    }

    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
        GlobalMockObject::verify();
        cout << "A Test case in RankGraph64Plus1Test TearDown" << endl;
    }
};

TEST_F(RankGraph64Plus1Test, Ut_GetPeer2PlaneEdges_When_MultiPortNoiseNotInLayer0_Expect_Ignored)
{
    auto phyTopoGraph = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
    auto phyPeer = std::make_shared<PhyTopo::Peer>(64);
    auto phyFabric = std::make_shared<PhyTopo::Fabric>();
    const NodeId peerNodeId = PhyTopo::Peer::GetId(64);
    const NodeId fabricNodeId = PhyTopo::Fabric::GetId();
    phyTopoGraph->AddNode(peerNodeId, phyPeer);
    phyTopoGraph->AddNode(fabricNodeId, phyFabric);

    auto rankPeer = std::make_shared<NetInstance::Peer>(0, 64, 64, 0);
    const std::vector<std::set<std::string>> physicalPlanePortGroups = {
        {"0/0", "0/1", "0/2", "0/3"},
        {"0/4", "0/5", "0/6", "0/7"},
        {"1/0", "1/1", "1/2", "1/3"},
        {"1/4", "1/5", "1/6", "1/7"},
    };
    std::map<std::string, std::vector<IpAddress>> layer0PortAddrMap;
    layer0PortAddrMap["1/0"] = {IpAddress("192.168.0.1")};
    rankPeer->SetPortPortAddrMapLayer0(layer0PortAddrMap);

    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = LinkType::PEER2NET;
    linkAttrs.protocols = {LinkProtocol::UB_CTP};
    for (u32 planeId = 0; planeId < static_cast<u32>(physicalPlanePortGroups.size()); ++planeId) {
        auto edge = std::make_shared<PhyTopo::Link>(phyPeer, phyFabric, linkAttrs, TopoType::CLOS, planeId);
        edge->SetSourceIface(std::make_shared<PhyTopo::ConnInterface>(
            physicalPlanePortGroups[planeId], AddrPosition::DEVICE, LinkType::PEER2NET, linkAttrs.protocols));
        phyTopoGraph->AddEdge(peerNodeId, fabricNodeId, edge);
    }

    auto noiseEdge = std::make_shared<PhyTopo::Link>(phyPeer, phyFabric, linkAttrs, TopoType::CLOS, 2);
    noiseEdge->SetSourceIface(std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"9/0", "9/1", "9/2", "9/3"}, AddrPosition::DEVICE, LinkType::PEER2NET,
        linkAttrs.protocols));
    phyTopoGraph->AddEdge(peerNodeId, fabricNodeId, noiseEdge);

    UpdaterFor64Plus1 updater;
    std::shared_ptr<PhyTopo::Link> matchedEdge;
    EXPECT_NO_THROW(
        matchedEdge
        = updater.GetPeer2PlaneEdges(2, rankPeer, phyTopoGraph, BACKUP_TO_PLANE_ADDR_NUM, linkAttrs.protocols));
    ASSERT_NE(matchedEdge, nullptr);
    EXPECT_EQ(matchedEdge->GetTopoInstId(), 2);
}

TEST_F(RankGraph64Plus1Test, test_4p_without_backup)
{
    // ranktable不使用备份, topo文件也无备份信息
    RankGraphBuilder rankGraphBuilder;
    unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable4p64Plus1(false), test::MakeFourPeerMeshTopo(), 0);

    EXPECT_NE(rankGraph, nullptr);
    // check innerRanks
    set<RankId> expectRanks{0, 1, 2, 3};
    EXPECT_EQ(rankGraph->innerRanks_, expectRanks);
    // check peers
    EXPECT_EQ(rankGraph->peers_.size(), 4);
    for (u32 i = 0; i < expectRanks.size(); i++) {
        EXPECT_EQ(rankGraph->peers_[i]->GetRankId(), i);
        EXPECT_EQ(rankGraph->peers_[i]->GetLocalId(), i);
        EXPECT_EQ(rankGraph->peers_[i]->GetNodeId(), i);
        EXPECT_EQ(rankGraph->peers_[i]->GetLevels().size(), 1);
    }
    // check GetPaths()
    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        NetInstance::Link link_0 = rankGraph->GetPaths(0, 0, i)[0].links[0];
        NetInstance::Link link_1 = rankGraph->GetPaths(0, i, 3)[0].links[0];
        EXPECT_EQ(link_0.source_->GetNodeId(), 0);
        EXPECT_EQ(link_0.target_->GetNodeId(), i);
        EXPECT_EQ(link_1.source_->GetNodeId(), i);
        EXPECT_EQ(link_1.target_->GetNodeId(), 3);
    }
    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        EXPECT_EQ(rankGraph->GetPaths(0, 0, i)[0].links[0].source_->GetNodeId(), 0);
        EXPECT_EQ(rankGraph->GetPaths(0, i, 0)[0].links[0].source_->GetNodeId(), i);
    }
}

TEST_F(RankGraph64Plus1Test, test_RankGraph_Build_should_failed_when_topo_missing_backup_edge)
{
    // 使用备份D但topo文件中缺少备份相关信息
    RankGraphBuilder rankGraphBuilder;
    EXPECT_THROW(
        rankGraphBuilder.RecoverBuild(test::MakeRankTable4p64Plus1(true), test::MakeFourPeerMeshTopo(), 0),
        InvalidParamsException);
}

TEST_F(RankGraph64Plus1Test, test_RankGraph_Build_without_Backup)
{
    // 直接启动，不使用备份D
    RankGraphBuilder rankGraphBuilder;
    unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable2x2(false), test::MakeTwoByTwoPlusBackupTopo(), 0);

    EXPECT_NE(rankGraph, nullptr);
    // check innerRanks
    set<RankId> expectRanks{0, 1, 2, 3};
    EXPECT_EQ(rankGraph->innerRanks_, expectRanks);
    // check peers
    EXPECT_EQ(rankGraph->peers_.size(), 4);
    EXPECT_EQ(rankGraph->peers_[0]->GetLocalId(), 0);
    EXPECT_EQ(rankGraph->peers_[1]->GetLocalId(), 1);
    EXPECT_EQ(rankGraph->peers_[2]->GetLocalId(), 8);
    EXPECT_EQ(rankGraph->peers_[3]->GetLocalId(), 9);
    for (u32 i = 0; i < expectRanks.size(); i++) {
        EXPECT_EQ(rankGraph->peers_[i]->GetRankId(), i);
        EXPECT_EQ(rankGraph->peers_[i]->GetNodeId(), i);
        EXPECT_EQ(rankGraph->peers_[i]->GetLevels().size(), 2);
    }
    // topo 中 net_layer=99 的 0/9 噪声端口不能污染 rankTable 定义的 layer 0 接口
    for (const auto& rankPeer : rankGraph->peers_) {
        const auto layer0Ifaces = rankPeer.second->GetIfacesByLayer(0);
        EXPECT_FALSE(layer0Ifaces.empty());
        for (const auto& iface : layer0Ifaces) {
            ASSERT_NE(nullptr, iface);
            EXPECT_EQ(0, iface->GetPorts().count("0/9"));
        }
    }
    // check fabGroups
    // check level0 az0-rack0
    NetInstance* netInstL0 = rankGraph->GetNetInstanceByNetInstId(0, "az0-rack0");
    EXPECT_NE(netInstL0, nullptr);
    EXPECT_EQ(netInstL0->rankIds, expectRanks);
    EXPECT_EQ(netInstL0->peers.size(), 4);
    ASSERT_EQ(netInstL0->fabrics.size(), 1);
    EXPECT_EQ(netInstL0->vGraph.nodes.size(), 5);
    EXPECT_EQ(netInstL0->fabrics[0]->GetPlaneId(), "0");

    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        EXPECT_EQ(netInstL0->vGraph.edges[0][i][0]->source_->GetNodeId(), 0);
        EXPECT_EQ(netInstL0->vGraph.edges[0][i][0]->target_->GetNodeId(), i);
        EXPECT_EQ(netInstL0->vGraph.edges[i][3][0]->source_->GetNodeId(), i);
        EXPECT_EQ(netInstL0->vGraph.edges[i][3][0]->target_->GetNodeId(), 3);
    }
    // check level1 az0-level1
    NetInstance* netInstL1 = rankGraph->GetNetInstanceByNetInstId(1, "az0-layer1");
    EXPECT_NE(netInstL1, nullptr);
    EXPECT_EQ(netInstL1->rankIds, expectRanks);
    EXPECT_EQ(netInstL1->peers.size(), 4);
    EXPECT_EQ(netInstL1->fabrics.size(), 1);
    EXPECT_EQ(netInstL1->vGraph.nodes.size(), 5);
    // check GetPaths()
    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        EXPECT_EQ(rankGraph->GetPaths(0, 0, i)[0].links[0].source_->GetNodeId(), 0);
        EXPECT_EQ(rankGraph->GetPaths(0, i, 0)[0].links[0].source_->GetNodeId(), i);
    }
}

TEST_F(RankGraph64Plus1Test, test_RankGraph_Build_with_Backup)
{
    // 直接启动，使用备份D
    RankGraphBuilder rankGraphBuilder;
    unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable2x2(true), test::MakeTwoByTwoPlusBackupTopo(), 0);

    EXPECT_NE(rankGraph, nullptr);
    // check innerRanks
    set<RankId> expectRanks{0, 1, 2, 3};
    EXPECT_EQ(rankGraph->innerRanks_, expectRanks);
    // check peers
    EXPECT_EQ(rankGraph->peers_.size(), 4);
    EXPECT_EQ(rankGraph->peers_[0]->GetLocalId(), 0);
    EXPECT_EQ(rankGraph->peers_[1]->GetLocalId(), 64);
    EXPECT_EQ(rankGraph->peers_[2]->GetLocalId(), 8);
    EXPECT_EQ(rankGraph->peers_[3]->GetLocalId(), 9);
    // check fabGroups
    // check level0 az0-rack0
    NetInstance* netInstL0 = rankGraph->GetNetInstanceByNetInstId(0, "az0-rack0");
    EXPECT_NE(netInstL0, nullptr);
    EXPECT_EQ(netInstL0->rankIds, expectRanks);
    EXPECT_EQ(netInstL0->peers.size(), 4);
    EXPECT_EQ(netInstL0->fabrics.size(), 1); // rankTable未配置plane_id，所有地址归入默认plane 0
    EXPECT_EQ(netInstL0->vGraph.nodes.size(), 5);
    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        EXPECT_EQ(netInstL0->vGraph.edges[0][i][0]->source_->GetNodeId(), 0);
        EXPECT_EQ(netInstL0->vGraph.edges[0][i][0]->target_->GetNodeId(), i);
        EXPECT_EQ(netInstL0->vGraph.edges[i][3][0]->source_->GetNodeId(), i);
        EXPECT_EQ(netInstL0->vGraph.edges[i][3][0]->target_->GetNodeId(), 3);
    }
    // check level1 az0-level1
    NetInstance* netInstL1 = rankGraph->GetNetInstanceByNetInstId(1, "az0-layer1");
    EXPECT_NE(netInstL1, nullptr);
    EXPECT_EQ(netInstL1->rankIds, expectRanks);
    EXPECT_EQ(netInstL1->peers.size(), 4);
    EXPECT_EQ(netInstL1->fabrics.size(), 1);
    EXPECT_EQ(netInstL1->vGraph.nodes.size(), 5);
    // check GetPaths()
    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        EXPECT_EQ(rankGraph->GetPaths(0, 0, i)[0].links[0].source_->GetNodeId(), 0);
        EXPECT_EQ(rankGraph->GetPaths(0, i, 0)[0].links[0].source_->GetNodeId(), i);
    }
}

TEST_F(RankGraph64Plus1Test, test_checkpoint_normal_to_backup)
{
    RankTableInfo rankTableInfo = test::MakeRankTable2x2(true);
    TopoInfo topoInfo = test::MakeTwoByTwoPlusBackupTopo();

    RankGraphBuilder rankGraphBuilder;
    unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0);

    EXPECT_NE(rankGraph, nullptr);
    // check innerRanks
    set<RankId> expectRanks{0, 1, 2, 3};
    EXPECT_EQ(rankGraph->innerRanks_, expectRanks);
    // check peers
    EXPECT_EQ(rankGraph->peers_.size(), 4);
    EXPECT_EQ(rankGraph->peers_[0]->GetLocalId(), 0);
    EXPECT_EQ(rankGraph->peers_[1]->GetLocalId(), 64);
    EXPECT_EQ(rankGraph->peers_[2]->GetLocalId(), 8);
    EXPECT_EQ(rankGraph->peers_[3]->GetLocalId(), 9);
    // check fabGroups
    // check level0 az0-rack0
    NetInstance* netInstL0 = rankGraph->GetNetInstanceByNetInstId(0, "az0-rack0");
    EXPECT_NE(netInstL0, nullptr);
    EXPECT_EQ(netInstL0->rankIds, expectRanks);
    EXPECT_EQ(netInstL0->peers.size(), 4);
    EXPECT_EQ(netInstL0->fabrics.size(), 1);
    EXPECT_EQ(netInstL0->vGraph.nodes.size(), 5);
    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        EXPECT_EQ(netInstL0->vGraph.edges[0][i][0]->source_->GetNodeId(), 0);
        EXPECT_EQ(netInstL0->vGraph.edges[0][i][0]->target_->GetNodeId(), i);
        EXPECT_EQ(netInstL0->vGraph.edges[i][3][0]->source_->GetNodeId(), i);
        EXPECT_EQ(netInstL0->vGraph.edges[i][3][0]->target_->GetNodeId(), 3);
    }
    // check level1 az0-level1
    NetInstance* netInstL1 = rankGraph->GetNetInstanceByNetInstId(1, "az0-layer1");
    EXPECT_NE(netInstL1, nullptr);
    EXPECT_EQ(netInstL1->rankIds, expectRanks);
    EXPECT_EQ(netInstL1->peers.size(), 4);
    EXPECT_EQ(netInstL1->fabrics.size(), 1);
    EXPECT_EQ(netInstL1->vGraph.nodes.size(), 5);
    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        EXPECT_EQ(rankGraph->GetPaths(0, 0, i)[0].links[0].source_->GetNodeId(), 0);
        EXPECT_EQ(rankGraph->GetPaths(0, i, 0)[0].links[0].source_->GetNodeId(), i);
    }
}

TEST_F(RankGraph64Plus1Test, test_checkpoint_normal_switch_pod_without_backup)
{
    RankTableInfo rankTableInfo = test::MakeRankTable2x2(false);
    TopoInfo topoInfo = test::MakeTwoByTwoPlusBackupTopo();

    RankGraphBuilder rankGraphBuilder;
    unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0);

    EXPECT_NE(rankGraph, nullptr);
    // check innerRanks
    set<RankId> expectRanks{0, 1, 2, 3};
    EXPECT_EQ(rankGraph->innerRanks_, expectRanks);
    // check peers
    EXPECT_EQ(rankGraph->peers_.size(), 4);
    EXPECT_EQ(rankGraph->peers_[0]->GetLocalId(), 0);
    EXPECT_EQ(rankGraph->peers_[1]->GetLocalId(), 1);
    EXPECT_EQ(rankGraph->peers_[2]->GetLocalId(), 8);
    EXPECT_EQ(rankGraph->peers_[3]->GetLocalId(), 9);
    // check fabGroups
    // check level0 az0-rack0
    NetInstance* netInstL0 = rankGraph->GetNetInstanceByNetInstId(0, "az0-rack0");
    EXPECT_NE(netInstL0, nullptr);
    EXPECT_EQ(netInstL0->rankIds, expectRanks);
    EXPECT_EQ(netInstL0->peers.size(), 4);
    EXPECT_EQ(netInstL0->fabrics.size(), 1);
    EXPECT_EQ(netInstL0->vGraph.nodes.size(), 5);
    for (u32 i = 1; i < expectRanks.size() - 1; i++) {
        EXPECT_EQ(netInstL0->vGraph.edges[0][i][0]->source_->GetNodeId(), 0);
        EXPECT_EQ(netInstL0->vGraph.edges[0][i][0]->target_->GetNodeId(), i);
        EXPECT_EQ(netInstL0->vGraph.edges[i][3][0]->source_->GetNodeId(), i);
        EXPECT_EQ(netInstL0->vGraph.edges[i][3][0]->target_->GetNodeId(), 3);
    }
    // check level1 az0-level1
    NetInstance* netInstL1 = rankGraph->GetNetInstanceByNetInstId(1, "az0-layer1");
    EXPECT_NE(netInstL1, nullptr);
    EXPECT_EQ(netInstL1->rankIds, expectRanks);
    EXPECT_EQ(netInstL1->peers.size(), 4);
    EXPECT_EQ(netInstL1->fabrics.size(), 1);
    EXPECT_EQ(netInstL1->vGraph.nodes.size(), 5);
}
