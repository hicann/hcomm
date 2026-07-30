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
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "invalid_params_exception.h"
#include "json_parser.h"
#include "types.h"

#define private public
#include "phy_topo_builder.h"
#include "topo_info.h"

#undef private

using namespace Hccl;

std::shared_ptr<TopoInfo> LoadTopoInfoStub(PhyTopoBuilder *This, const std::string &topoPath)
{
    std::string topoString = R"(
    {
    "version": "2.0",
    "peer_count" : 3,
    "peer_list" :[
        { "local_id" : 0},
        { "local_id" : 1},
        { "local_id" : 2}
    ],
    "edge_count" : 4,
    "edge_list": [
        {
            "net_layer": 0,
            "link_type": "PEER2PEER",
            "protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
            "local_a": 0,
            "local_a_ports": ["0/0"],
            "local_b": 1,
            "local_b_ports": ["0/1"],
            "position": "DEVICE"
        },
        {
            "net_layer": 0,
            "link_type": "PEER2PEER",
            "protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
            "local_a": 1,
            "local_a_ports": ["0/1"],
            "local_b": 2,
            "local_b_ports": ["0/2"],
            "position": "DEVICE"
        },
        {
            "net_layer": 0,
            "link_type": "PEER2NET",
            "protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
            "local_a": 1,
            "local_a_ports": ["0/1"],
            "position": "DEVICE"
        },
        {
            "net_layer": 1,
            "link_type": "PEER2NET",
            "protocols": ["UB_MEM"],
            "topo_type": "1DMESH",
            "topo_instance_id": 1,
            "local_a": 0,
            "local_a_ports": ["1/3", "1/4", "0/3", "0/4"],
            "local_b": null,
            "local_b_ports": null,
            "position": "DEVICE"
        }
    ]
    }
    )";
    std::shared_ptr<TopoInfo> topoInfo = std::make_shared<TopoInfo>();
    JsonParser topoParser;
    topoParser.ParseString(topoString, *topoInfo);
    return topoInfo;
}

std::shared_ptr<TopoInfo> LoadTopoInfoWithDiffProtocols(PhyTopoBuilder *This, const std::string &topoPath)
{
    std::string topoString = R"(
    {
    "version": "2.0",
    "peer_count" : 3,
    "peer_list" :[
        { "local_id" : 0},
        { "local_id" : 1},
        { "local_id" : 2}
    ],
    "edge_count" : 4,
    "edge_list": [
        {
            "net_layer": 0,
            "link_type": "PEER2PEER",
            "protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
            "local_a": 1,
            "local_a_ports": ["0/1"],
            "local_b": 2,
            "local_b_ports": ["0/2"],
            "position": "DEVICE"
        },
		{
            "net_layer": 0,
            "link_type": "PEER2PEER",
            "protocols": ["UB_MEM"],
            "local_a": 1,
            "local_a_ports": ["0/1"],
            "local_b": 0,
            "local_b_ports": ["0/1"],
            "position": "DEVICE"
        },
        {
            "net_layer": 0,
            "link_type": "PEER2NET",
            "protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
            "local_a": 0,
            "local_a_ports": ["0/1"],
            "position": "DEVICE"
        },
         {
            "net_layer": 0,
            "link_type": "PEER2NET",
            "protocols": ["TCP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
            "local_a": 1,
            "local_a_ports": ["0/1"],
            "position": "DEVICE"
        }
    ]
    }
    )";
    std::shared_ptr<TopoInfo> topoInfo = std::make_shared<TopoInfo>();
    JsonParser topoParser;
    topoParser.ParseString(topoString, *topoInfo);
    return topoInfo;
}

std::shared_ptr<TopoInfo> LoadTopoInfoWithRepeatEdge(PhyTopoBuilder *This, const std::string &topoPath)
{
    std::string topoString = R"(
    {
        "version": "2.0",
        "peer_count": 2,
        "peer_list": [
            { "local_id": 0 },
            { "local_id": 1 }
        ],
        "edge_count": 2,
        "edge_list": [
            {
                "net_layer": 0,
                "link_type": "PEER2PEER",
                "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
                "local_a": 0,
                "local_a_ports": ["0/1"],
                "local_b": 1,
                "local_b_ports": ["0/2"],
                "position": "DEVICE"
            },
            {
                "net_layer": 0,
                "link_type": "PEER2PEER",
                "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
                "local_a": 0,
                "local_a_ports": ["0/1"],
                "local_b": 1,
                "local_b_ports": ["0/2"],
                "position": "DEVICE"
            }
        ]
    }
    )";

    std::shared_ptr<TopoInfo> topoInfo = std::make_shared<TopoInfo>();
    JsonParser topoParser;
    topoParser.ParseString(topoString, *topoInfo);
    return topoInfo;
}

std::unique_ptr<PhyTopo> PhyTopoBuilderBuildStub(const std::string &topoPath)
{
    if (topoPath.empty()) {
        THROW<InvalidParamsException>("[PhyTopoBuilder::%s]Topo path is empty.", __func__);
    }

    HCCL_DEBUG("[PhyTopoBuilder::%s]Start to build physic topo.", __func__);
    std::unique_ptr<PhyTopo> phyTopo = std::make_unique<PhyTopo>();
    PhyTopoBuilder phyTopoBuilder;
    auto topoInfo = phyTopoBuilder.LoadTopoInfo(topoPath);
    // topoInfo 中的全部物理边构造成一张 Graph。
    auto graph = phyTopoBuilder.CreateGraph(topoInfo->edges);
    phyTopo->AddTopoGraph(graph);
    return phyTopo;
}

class PhyTopoBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PhyTopoBuilderTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PhyTopoBuilderTest TearDown" << std::endl;
    }

    virtual void SetUp()
    {
        PhyTopo::GetInstance()->Clear();
        std::cout << "A Test case in PhyTopoBuilderTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
        GlobalMockObject::verify();
        std::cout << "A Test case in PhyTopoBuilderTest TearDown" << std::endl;
    }
};

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_EmptyTopoPath_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "";
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_InValidTopoPath_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "../topo.json";
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_MaxTopoSizeFile_Expect_ThrowInvalidParamsException)
{
    std::string topoPath{HCOMM_CODE_ROOT_DIR "test/legacy/ut/framework/topo/new_topo_builder/phy_topo_builder/largeTopo.json"};
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_ErrorFilePath_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "llt/ace/comop/hccl/noneFile.json";
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_ValidTopoPath_Expect_ReturnEdgeNum)
{
    std::string topoPath = "./topo.json";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoStub));
    std::unique_ptr<PhyTopo> phyTopo = PhyTopoBuilderBuildStub(topoPath);
    auto graph = phyTopo->GetTopoGraph();
    size_t totalEdgeNum = 0;

    // 遍历所有源节点
    for (const auto &srcEntry : graph->edges) {
        const auto &dstMap = srcEntry.second;
        // 遍历该源节点下的所有目标节点
        for (const auto &dstEntry : dstMap) {
            const auto &edgesVec = dstEntry.second;  // 该源->目标的所有边
            totalEdgeNum += edgesVec.size();
        }
    }

    EXPECT_EQ(totalEdgeNum, 8);
}


TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_EdgeRepeat_Expect_MergeEdge)
{
    std::string topoPath = "./test_topo.json";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoWithRepeatEdge));
    std::unique_ptr<PhyTopo> phyTopo;
    EXPECT_NO_THROW(phyTopo = PhyTopoBuilderBuildStub(topoPath));
    ASSERT_NE(phyTopo, nullptr);
    auto graph = phyTopo->GetTopoGraph();
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->GetEdges(PhyTopo::Peer::GetId(0), PhyTopo::Peer::GetId(1)).size(), 1);
    EXPECT_EQ(graph->GetEdges(PhyTopo::Peer::GetId(1), PhyTopo::Peer::GetId(0)).size(), 1);
}

TEST_F(PhyTopoBuilderTest, Ut_BuildPhyTopo_When_NoEdges_Expect_KeepIsolatedPeers)
{
    TopoInfo topoInfo;
    topoInfo.version = "2.0";
    topoInfo.peerCount = 3;
    for (u32 localId = 0; localId < topoInfo.peerCount; ++localId) {
        PeerInfo peer;
        peer.localId = localId;
        topoInfo.peers.emplace_back(peer);
    }

    PhyTopoBuilder phyTopoBuilder;
    phyTopoBuilder.BuildPhyTopo(topoInfo);

    auto graph = PhyTopo::GetInstance()->GetTopoGraph();
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->nodes.size(), 3);
    EXPECT_TRUE(graph->edges.empty());
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_DiffProtocols_Expect_ReturnEdgeNum)
{
    std::string topoPath = "./topo.json";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoWithDiffProtocols));
    std::unique_ptr<PhyTopo> phyTopo = PhyTopoBuilderBuildStub(topoPath);
    auto graph = phyTopo->GetTopoGraph();
    size_t totalEdgeNum = 0;

    // 遍历所有源节点
    for (const auto &srcEntry : graph->edges) {
        const auto &dstMap = srcEntry.second;
        // 遍历该源节点下的所有目标节点
        for (const auto &dstEntry : dstMap) {
            const auto &edgesVec = dstEntry.second;  // 该源->目标的所有边
            totalEdgeNum += edgesVec.size();
        }
    }
    EXPECT_EQ(totalEdgeNum, 8);

}