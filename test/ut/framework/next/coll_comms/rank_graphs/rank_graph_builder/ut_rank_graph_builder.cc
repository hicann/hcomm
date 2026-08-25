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

#include <initializer_list>
#include <iostream>
#include <set>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include <hccl/hccl_types.h>

#include "hccp_ctx.h"
#include "rdma_handle_manager.h"

namespace hcomm {
HcclResult HccpRaGetDevBaseAttr(void* ctxHandle, struct DevBaseAttr* attr);
}

#define private public
#define protected public

#include "detour_service.h"
#include "orion_adapter_hccp.h"
#include "phy_topo.h"
#include "phy_topo_builder.h"
#include "rank_graph_test_data_builder.h"
#include "rank_gph.h"
#include "rank_graph_builder.h"

#undef private
#undef protected

using namespace Hccl;

namespace {
AddressInfo MakeEidAddress(const std::string& eid, std::initializer_list<std::string> ports, const std::string& planeId)
{
    AddressInfo address;
    address.addr = IpAddress(IpAddress::StrToEID(eid));
    address.addrType = AddrType::EID;
    address.ports = std::set<std::string>(ports.begin(), ports.end());
    address.planeId = planeId;
    return address;
}

std::vector<AddressInfo> MakeOverlappedUbAddresses(u32 rankId, LinkProtocol protocol)
{
    const std::string eid0 = rankId == 0U ? "00000000000000000000000000000001" : "00000000000000000000000000000011";
    const std::string eid1 = rankId == 0U ? "00000000000000000000000000000002" : "00000000000000000000000000000012";
    if (protocol == LinkProtocol::UB_CTP) {
        return {
            MakeEidAddress(eid0, {"0/0", "0/1", "0/2", "0/3", "0/4", "0/5"}, "planeA"),
            MakeEidAddress(eid1, {"0/6", "0/7"}, "planeB"),
        };
    }
    return {
        MakeEidAddress(eid0, {"0/0", "0/1"}, "planeA"),
        MakeEidAddress(eid1, {"0/6", "0/7"}, "planeB"),
    };
}

RankTableInfo MakeOverlappedUbRankTable(u32 netLayer, LinkProtocol protocol)
{
    return test::MakeRankTable({
        test::MakeRankInfo(
            0, 0, 0,
            {
                test::MakeRankLevel(
                    0, "inner", NetType::TOPO_FILE_DESC,
                    {
                        test::MakeAddress("192.168.0.1", {"1/0"}),
                    }),
                test::MakeRankLevel(netLayer, "all", NetType::CLOS, MakeOverlappedUbAddresses(0, protocol)),
            }),
        test::MakeRankInfo(
            1, 1, 1,
            {
                test::MakeRankLevel(
                    0, "inner", NetType::TOPO_FILE_DESC,
                    {
                        test::MakeAddress("192.168.0.2", {"1/1"}),
                    }),
                test::MakeRankLevel(netLayer, "all", NetType::CLOS, MakeOverlappedUbAddresses(1, protocol)),
            }),
    });
}

RankTableInfo MakeOverlappedUbRankTableForLayer1And2()
{
    RankTableInfo rankTable = MakeOverlappedUbRankTable(1, LinkProtocol::UB_CTP);
    for (auto& rankInfo : rankTable.ranks) {
        rankInfo.rankLevelInfos.emplace_back(test::MakeRankLevel(
            2, "all-layer2", NetType::CLOS, MakeOverlappedUbAddresses(rankInfo.rankId, LinkProtocol::UB_CTP)));
    }
    return rankTable;
}

TopoInfo MakeOverlappedUbTopo(bool addLocalTpEdge = true, bool addDuplicateCtpEdge = false)
{
    std::vector<EdgeInfo> edges = {
        test::MakeEdge(0, LinkType::PEER2PEER, 0, {"1/0"}, 1, {"1/1"}),
    };
    for (u32 localId = 0U; localId < 2U; ++localId) {
        EdgeInfo ctpEdge
            = test::MakeEdge(0, LinkType::PEER2NET, localId, {"0/0", "0/1", "0/2", "0/3", "0/4", "0/5", "0/6", "0/7"});
        ctpEdge.protocols = {LinkProtocol::UB_CTP, LinkProtocol::UB_MEM};
        if (addDuplicateCtpEdge) {
            EdgeInfo duplicateCtpEdge = ctpEdge;
            duplicateCtpEdge.topoInstId = 8;
            edges.emplace_back(std::move(duplicateCtpEdge));
        }
        edges.emplace_back(std::move(ctpEdge));
        if (addLocalTpEdge || localId != 0U) {
            edges.emplace_back(test::MakeEdge(
                0, LinkType::PEER2NET, localId, {"0/0", "0/1", "0/6", "0/7"}, 0, {}, LinkProtocol::UB_TP));
        }
    }
    return test::MakeTopoInfo({0, 1}, edges);
}

void MockDevBaseAttr(DevBaseAttr* devBaseAttr, u32 callCount = 2U)
{
    MOCKER_CPP(&RdmaHandleManager::GetByIp).stubs().will(returnValue(static_cast<RdmaHandle>(devBaseAttr)));
    MOCKER(hcomm::HccpRaGetDevBaseAttr)
        .expects(exactly(callCount))
        .with(mockcpp::any(), outBoundP(devBaseAttr, sizeof(*devBaseAttr)))
        .will(returnValue(HCCL_SUCCESS));
}

void CheckOverlappedUbPaths(
    const RankGraph& rankGraph, u32 netLayer, const std::set<LinkProtocol>& expectedProtocols,
    const std::multiset<size_t>& expectedPortCounts)
{
    const std::vector<NetInstance::Path> paths = rankGraph.GetPaths(netLayer, 0, 1);
    ASSERT_EQ(2, paths.size());
    std::multiset<size_t> portCounts;
    for (const auto& path : paths) {
        ASSERT_EQ(2, path.links.size());
        EXPECT_EQ(expectedProtocols, path.links[0].GetLinkProtocols());
        EXPECT_EQ(expectedProtocols, path.links[1].GetLinkProtocols());
        ASSERT_NE(nullptr, path.links[0].GetSourceIface());
        portCounts.emplace(path.links[0].GetSourceIface()->GetPorts().size());
    }
    EXPECT_EQ(expectedPortCounts, portCounts);
}

void CheckSingleTopoInst(const RankGraph& rankGraph, u32 netLayer)
{
    const NetInstance* netInst = rankGraph.GetNetInstanceByRankId(netLayer, 0);
    ASSERT_NE(nullptr, netInst);
    std::vector<u32> topoInstIds;
    u32 topoInstNum = 0U;
    netInst->GetTopoInstsByLayer(topoInstIds, topoInstNum);
    EXPECT_EQ(1U, topoInstNum);
    EXPECT_EQ((std::vector<u32>{0U}), topoInstIds);
}

constexpr char SAME_TOPO_INST_RANK_TABLE[] = R"({
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

constexpr char SAME_TOPO_INST_TOPO[] = R"({
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
} // namespace

class RankGraphBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankGraphBuilderTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankGraphBuilderTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankGraphBuilderTest SetUP" << std::endl;
        GlobalMockObject::reset();
        PhyTopo::GetInstance()->Clear();
    }

    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in RankGraphBuilderTest TearDown" << std::endl;
    }
};

TEST_F(RankGraphBuilderTest, Ut_Build_When_Normal_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    TopoInfo topoInfo = test::MakeTwoPeerClosTopo();
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(test::MakeRankTable2p(), topoInfo, 0);
    EXPECT_NE(nullptr, rankGraph);
    ASSERT_NE(nullptr, rankGraphBuilder.topoInfo_);
    EXPECT_EQ(topoInfo.Describe(), rankGraphBuilder.topoInfo_->Describe());
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
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable1p(), test::MakeOnePeerTopo(true), 0);
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
    auto graph = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
    graph->AddNode(PhyTopo::Peer::GetId(0), std::make_shared<PhyTopo::Peer>(0));
    PhyTopo::GetInstance()->AddTopoGraph(graph);
    PhyTopo::GetInstance()->InitFinish();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable1p(), test::MakeOnePeerTopo(false), 0);
    EXPECT_NE(nullptr, rankGraph);
    auto rankSize = rankGraph->GetRankSize();
    EXPECT_EQ(rankSize, 1);
    auto peer = rankGraph->GetPeer(rankGraph->GetMyRank());
    ASSERT_NE(peer, nullptr);
    EXPECT_EQ(peer->GetLocalId(), 0);
}

TEST_F(RankGraphBuilderTest, Ut_BuildFromRankTable_When_NetLayerMissingInTopo_Expect_Success)
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
    RankTableInfo rankTable = test::MakeRankTable2p();
    rankTable.ranks[0].rankLevelInfos.emplace_back(test::MakeRankLevel(
        2, "missing-layer", NetType::CLOS,
        {
            test::MakeAddress("192.168.200.1", {"0/1"}),
        }));
    std::unique_ptr<RankGraph> rankGraph;
    EXPECT_NO_THROW(rankGraph = rankGraphBuilder.Build(rankTable, "topo.json", 0));
    EXPECT_NE(rankGraph, nullptr);
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_TopoWithoutNetLayer_Expect_UseRankTableLayers)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable4pForBuilder(), test::MakeFourPeerBuilderTopo(), 0);
    ASSERT_NE(nullptr, rankGraph);
    const std::set<u32> expectedLevels = {0, 1, 2};
    EXPECT_EQ(3, rankGraph->GetLevelNum());
    EXPECT_EQ(expectedLevels, rankGraph->GetLevels(0));
    std::vector<std::string> netIds = {"az0-rack0", "az0", "all"};
    for (s32 rankId = 0; rankId < 4; rankId++) {
        for (u32 netLayer = 0; netLayer < 3; netLayer++) {
            const NetInstance* fabGroup = rankGraph->GetNetInstanceByRankId(netLayer, rankId);
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

    const NetInstance* layer0 = rankGraph->GetNetInstanceByNetInstId(0, "az0-rack0");
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

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_EidSupportsCtpAndRtp_Expect_AllUbProtocols)
{
    DevBaseAttr devBaseAttr{};
    devBaseAttr.ub.priorityInfo[0].tpType.bs.ctp = 1;
    devBaseAttr.ub.priorityInfo[Hccl::kRaUbGetTpInfoParamDefaultQos].tpType.bs.rtp = 1;
    MockDevBaseAttr(&devBaseAttr);

    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(MakeOverlappedUbRankTable(1, LinkProtocol::UB_CTP), MakeOverlappedUbTopo(), 0);

    ASSERT_NE(nullptr, rankGraph);
    const std::set<u32> expectedLevels = {0, 1};
    EXPECT_EQ(expectedLevels, rankGraph->GetLevels(0));
    CheckOverlappedUbPaths(*rankGraph, 1, {LinkProtocol::UB_CTP, LinkProtocol::UB_TP, LinkProtocol::UB_MEM}, {2, 6});
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_Layer1And2HaveDuplicatePhyLinks_Expect_KeepOneTopoInst)
{
    DevBaseAttr devBaseAttr{};
    devBaseAttr.ub.priorityInfo[0].tpType.bs.ctp = 1;
    devBaseAttr.ub.priorityInfo[Hccl::kRaUbGetTpInfoParamDefaultQos].tpType.bs.rtp = 1;
    MockDevBaseAttr(&devBaseAttr, 4U);

    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(MakeOverlappedUbRankTableForLayer1And2(), MakeOverlappedUbTopo(true, true), 0);

    ASSERT_NE(nullptr, rankGraph);
    for (u32 netLayer : {1U, 2U}) {
        CheckSingleTopoInst(*rankGraph, netLayer);
        CheckOverlappedUbPaths(
            *rankGraph, netLayer, {LinkProtocol::UB_CTP, LinkProtocol::UB_TP, LinkProtocol::UB_MEM}, {2, 6});
    }
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_PortsOverlapAndEidsUseTp_Expect_TpLinks)
{
    DevBaseAttr devBaseAttr{};
    devBaseAttr.ub.priorityInfo[Hccl::kRaUbGetTpInfoParamDefaultQos].tpType.bs.rtp = 1;
    MockDevBaseAttr(&devBaseAttr);

    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(MakeOverlappedUbRankTable(2, LinkProtocol::UB_TP), MakeOverlappedUbTopo(), 0);

    ASSERT_NE(nullptr, rankGraph);
    CheckOverlappedUbPaths(*rankGraph, 2, {LinkProtocol::UB_TP}, {2, 2});
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_OnlyRemotePortsOverlapAndEidsUseCtp_Expect_IgnoreTpLinks)
{
    DevBaseAttr devBaseAttr{};
    devBaseAttr.ub.priorityInfo[Hccl::kRaUbGetTpInfoParamDefaultQos].tpType.bs.ctp = 1;
    MockDevBaseAttr(&devBaseAttr);

    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(
        MakeOverlappedUbRankTable(1, LinkProtocol::UB_CTP), MakeOverlappedUbTopo(false), 0);

    ASSERT_NE(nullptr, rankGraph);
    CheckOverlappedUbPaths(*rankGraph, 1, {LinkProtocol::UB_CTP, LinkProtocol::UB_MEM}, {2, 6});
}

TEST_F(RankGraphBuilderTest, Ut_RecoverBuild_When_SameTopoInstHasDifferentPlanes_Expect_ReuseFabricByTopoInstId)
{
    PhyTopo::GetInstance()->Clear();

    JsonParser parser;
    RankTableInfo rankTableInfo;
    TopoInfo topoInfo;
    parser.ParseString(SAME_TOPO_INST_RANK_TABLE, rankTableInfo);
    parser.ParseString(SAME_TOPO_INST_TOPO, topoInfo);

    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0);
    ASSERT_NE(nullptr, rankGraph);
    NetInstance* layer0 = rankGraph->GetNetInstanceByNetInstId(0, "rank-group-0");
    ASSERT_NE(nullptr, layer0);
    const auto& fabrics = layer0->GetFabrics();
    ASSERT_EQ(1, fabrics.size());
    ASSERT_NE(nullptr, fabrics.front());
    EXPECT_EQ((1ULL << 32) | 7ULL, fabrics.front()->GetNodeId());
}

TEST_F(RankGraphBuilderTest, Ut_ConstructConnI_When_PortMapEmpty_Expect_OnlyPcieCreatesD2hIface)
{
    const std::map<std::string, std::vector<IpAddress>> emptyPortAddrMap;
    auto pcieIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"0/0"}, AddrPosition::DEVICE, LinkType::PEER2PEER,
        std::set<LinkProtocol>{LinkProtocol::PCIE});
    MOCKER(HrtRaSocketGetVnicIpInfos).stubs();

    auto pcieNetIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(pcieIface, emptyPortAddrMap, TopoType::CLOS, 0, 0);

    ASSERT_EQ(1, pcieNetIfaces.size());
    EXPECT_EQ(std::set<std::string>{"d2h"}, pcieNetIfaces[0]->GetPorts());
    EXPECT_EQ(std::set<LinkProtocol>{LinkProtocol::PCIE}, pcieNetIfaces[0]->GetLinkProtocols());

    auto unmatchedIface = std::make_shared<PhyTopo::ConnInterface>(
        std::set<std::string>{"0/9"}, AddrPosition::DEVICE, LinkType::PEER2PEER,
        std::set<LinkProtocol>{LinkProtocol::UB_CTP});
    auto unmatchedNetIfaces
        = ConstructConnIFromPhyTopoConnIAndPortMap(unmatchedIface, emptyPortAddrMap, TopoType::CLOS, 0, 0);

    EXPECT_TRUE(unmatchedNetIfaces.empty());
}

TEST_F(RankGraphBuilderTest, Ut_RankGraphBuilderRecoverBuild_When_Invalid_Expect_InvalidParamsException)
{
    RankTableInfo rankTableInfo;
    TopoInfo topoInfo;
    RankGraphBuilder rankGraphBuilder;
    EXPECT_THROW(rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0), InvalidParamsException);
}
