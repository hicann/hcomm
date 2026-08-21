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
#include <nlohmann/json.hpp>
#include "topo_info.h"
#include "json_parser.h"
#include "orion_adapter_rts.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
using namespace Hccl;

namespace {
std::string BuildLargeTopoString()
{
    constexpr u32 peerCount = 64;
    nlohmann::json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardwareType"] = "Atlas 950 SuperPod 2D";
    topoJson["peer_count"] = peerCount;
    topoJson["edge_count"] = peerCount;
    topoJson["peer_list"] = nlohmann::json::array();
    topoJson["edge_list"] = nlohmann::json::array();

    for (u32 localId = 0; localId < peerCount; ++localId) {
        topoJson["peer_list"].push_back({{"local_id", localId}});
        topoJson["edge_list"].push_back({
            {"net_layer", 2},
            {"link_type", "PEER2NET"},
            {"topo_type", "CLOS"},
            {"topo_instance_id", 0},
            {"topo_attr", ""},
            {"local_a", localId},
            {"local_a_ports", {"0/1", "0/2", "1/1", "1/2"}},
            {"protocols", {"UB_TP"}},
            {"position", "DEVICE"},
        });
    }
    return topoJson.dump();
}
} // namespace

namespace {
void WriteLegacyEdge(BinaryStream& binaryStream, u32 binaryLayer, const EdgeInfo& edge)
{
    binaryStream << binaryLayer << static_cast<u32>(edge.linkType) << static_cast<u32>(edge.topoType)
                 << edge.topoInstId;
    binaryStream << edge.protocols.size();
    for (const auto& protocol : edge.protocols) {
        binaryStream << static_cast<u32>(protocol);
    }
    binaryStream << edge.localA << edge.localB;
    binaryStream << edge.localAPorts.size();
    for (const auto& port : edge.localAPorts) {
        binaryStream << port;
    }
    binaryStream << edge.localBPorts.size();
    for (const auto& port : edge.localBPorts) {
        binaryStream << port;
    }
    binaryStream << static_cast<u32>(edge.position);
}
} // namespace

class TopoParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopoParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "TopoParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in TopoParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in TopoParserTest TearDown" << std::endl;
    }
};

// 功能用例，PEER2NET的B端口缺省，topoType和topoInstId缺省，正常填写
TEST_F(TopoParserTest, Ut_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 5,
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
			    "protocols": ["UB_MEM"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 2,
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
			    "local_a_ports": ["0/0"],
			    "position": "HOST"
		    },
		    {
                "net_layer": 1,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_TP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 1,
			    "local_a_ports": ["0/0"],
			    "local_b": 2,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    },
		    {
                "net_layer": 2,
                "link_type": "PEER2NET",
			    "protocols": ["ROCE"],
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "position": "DEVICE"
		    }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    topoParser.ParseString(topoString, topoInfo);

    TopoInfo expectTopoInfo;
    expectTopoInfo.version = "2.0";
    expectTopoInfo.peerCount = 3;

    PeerInfo peer0;
    peer0.localId = 0;
    PeerInfo peer1;
    peer1.localId = 1;
    PeerInfo peer2;
    peer2.localId = 2;
    expectTopoInfo.peers.emplace_back(peer0);
    expectTopoInfo.peers.emplace_back(peer1);
    expectTopoInfo.peers.emplace_back(peer2);

    expectTopoInfo.edgeCount = 5;
    EdgeInfo edge0;
    edge0.linkType = LinkType::PEER2PEER;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/1");
    edge0.position = AddrPosition::DEVICE;
    expectTopoInfo.edges.emplace_back(edge0);

    EdgeInfo edge1;
    edge1.linkType = LinkType::PEER2PEER;
    edge1.protocols.emplace(LinkProtocol::UB_MEM);
    edge1.topoType = TopoType::MESH_1D;
    edge1.topoInstId = 0;
    edge1.localA = 0;
    edge1.localAPorts.emplace("0/0");
    edge1.localB = 2;
    edge1.localBPorts.emplace("0/1");
    edge1.position = AddrPosition::DEVICE;
    expectTopoInfo.edges.emplace_back(edge1);

    EdgeInfo edge2;
    edge2.linkType = LinkType::PEER2NET;
    edge2.protocols.emplace(LinkProtocol::UB_CTP);
    edge2.topoType = TopoType::MESH_1D;
    edge2.topoInstId = 0;
    edge2.localA = 0;
    edge2.localAPorts.emplace("0/0");
    edge2.position = AddrPosition::HOST;
    expectTopoInfo.edges.emplace_back(edge2);

    EdgeInfo edge3;
    edge3.linkType = LinkType::PEER2PEER;
    edge3.protocols.emplace(LinkProtocol::UB_TP);
    edge3.topoType = TopoType::MESH_1D;
    edge3.topoInstId = 0;
    edge3.localA = 1;
    edge3.localAPorts.emplace("0/0");
    edge3.localB = 2;
    edge3.localBPorts.emplace("0/1");
    edge3.position = AddrPosition::DEVICE;
    expectTopoInfo.edges.emplace_back(edge3);

    EdgeInfo edge4;
    edge4.linkType = LinkType::PEER2NET;
    edge4.protocols.emplace(LinkProtocol::ROCE);
    edge4.topoType = TopoType::CLOS;
    edge4.topoInstId = 0;
    edge4.localA = 0;
    edge4.localAPorts.emplace("0/0");
    edge4.position = AddrPosition::DEVICE;
    expectTopoInfo.edges.emplace_back(edge4);

    EXPECT_EQ(topoInfo.version, expectTopoInfo.version);
    EXPECT_EQ(topoInfo.peerCount, expectTopoInfo.peerCount);
    EXPECT_EQ(topoInfo.peers.size(), expectTopoInfo.peers.size());
    for (u32 i = 0; i < topoInfo.peers.size(); i++) {
        EXPECT_EQ(topoInfo.peers[i].localId, expectTopoInfo.peers[i].localId);
    }
    EXPECT_EQ(topoInfo.edgeCount, expectTopoInfo.edgeCount);
    EXPECT_EQ(topoInfo.edges.size(), expectTopoInfo.edges.size());

    EXPECT_EQ(topoInfo.edges, expectTopoInfo.edges);

    EXPECT_EQ(topoInfo.Describe(), expectTopoInfo.Describe());
}

// 有效字段缺少
TEST_F(TopoParserTest, Ut_Deserialize_When_NeededFieldMissing_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1"
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// version = 1.0
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidVersion_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "1.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "id" : 0 },
		    { "id" : 1 },
		    { "id" : 2 }
	    ],
	    "edge_count" : 1,
        "edge_list": [
		    {
			    "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 2,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 1,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// warning, edge 为 0
TEST_F(TopoParserTest, Ut_Deserialize_When_ZeroEdge_Expect_Warning)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 0,
        "edge_list": []
        })";
    JsonParser topoParser;
    TopoInfo topoInfo;
    topoParser.ParseString(topoString, topoInfo);

    TopoInfo expectTopoInfo;
    expectTopoInfo.version = "2.0";
    expectTopoInfo.peerCount = 3;

    PeerInfo peer0;
    peer0.localId = 0;
    PeerInfo peer1;
    peer1.localId = 1;
    PeerInfo peer2;
    peer2.localId = 2;
    expectTopoInfo.peers.emplace_back(peer0);
    expectTopoInfo.peers.emplace_back(peer1);
    expectTopoInfo.peers.emplace_back(peer2);

    expectTopoInfo.edgeCount = 0;

    EXPECT_EQ(topoInfo.version, expectTopoInfo.version);
    EXPECT_EQ(topoInfo.peerCount, expectTopoInfo.peerCount);
    EXPECT_EQ(topoInfo.peers.size(), expectTopoInfo.peers.size());
    for (u32 i = 0; i < topoInfo.peers.size(); i++) {
        EXPECT_EQ(topoInfo.peers[i].localId, expectTopoInfo.peers[i].localId);
    }
    EXPECT_EQ(topoInfo.edgeCount, expectTopoInfo.edgeCount);
    EXPECT_EQ(topoInfo.edges.size(), expectTopoInfo.edges.size());
}

// peer_count != peer_list.size()
TEST_F(TopoParserTest, Ut_Deserialize_When_PeersSizeUnequalToPeerCount_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "peer_count" : 10,
        "edge_count" : 0,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// peer.loadId >= peer_count
TEST_F(TopoParserTest, Ut_Deserialize_When_PeerIdGreaterThanPeerCount_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "peer_count" : 3,
		"edge_count" : 0,
        "peer_list" :[
		    { "local_id" : 4 },
		    { "local_id" : 0 },
		    { "local_id" : 2 }
	    ],
      "edge_list": []
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    topoParser.ParseString(topoString, topoInfo);
    EXPECT_EQ(topoInfo.version, "2.0");
    EXPECT_EQ(topoInfo.peerCount, 3);
    EXPECT_EQ(topoInfo.peers.size(), 3);
    EXPECT_EQ(topoInfo.edgeCount, 0);
    EXPECT_EQ(topoInfo.edges.size(), 0);
}

// 重复的peer
TEST_F(TopoParserTest, Ut_Deserialize_When_DuplicatePeer_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "peer_count" : 3,
		"edge_count" : 0,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 0 },
		    { "local_id" : 2 }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// edge_count != edge_list.size()
TEST_F(TopoParserTest, Ut_Deserialize_When_EdgesSizeUnequalToEdgeCount_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 5,
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
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 2,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    }
			]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// 重复的边 PEER2PEER，localA和localB对调
TEST_F(TopoParserTest, Ut_Deserialize_When_DuplicateEdge_Expect_Merged)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 2,
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
                "net_layer": 7,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 1,
			    "local_a_ports": ["0/1"],
			    "local_b": 0,
			    "local_b_ports": ["0/0"],
			    "position": "DEVICE"
		    }
			]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_NO_THROW(topoParser.ParseString(topoString, topoInfo));
    EXPECT_EQ(topoInfo.edgeCount, 1);
    ASSERT_EQ(topoInfo.edges.size(), 1);
    EXPECT_EQ(topoInfo.edges[0].localA, 0);
    EXPECT_EQ(topoInfo.edges[0].localB, 1);
}

// Endpoint的localId无效
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidEndpointLocalId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 2,
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
			    "local_b": 3,
			    "local_b_ports": ["0/0"],
			    "position": "DEVICE"
		    }
			]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// 可缺省字段填无效值 "MESH"
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidTopoType_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 2,
        "edge_list": [
		    {
                "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "MESH",
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
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 1,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    }
			]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// 无效的JSON文件
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidJson_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "net_layer": 0,
        "link_type": "PEER2PEER",
		"protocols": ["UB_CTP"
        "topo_type": "1DMESH",
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException); // ?????????
}

// 字符串输入非法值
//  ranktable对应

// 越界输入 -1; 9999999999999999999999999999999999
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidTopoInstId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 1,
        "edge_list": [
		    {
			    "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": -3,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 1,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

TEST_F(TopoParserTest, Ut_BinaryStream_When_GetBinStreamToReBuild_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    TopoInfo expectTopoInfo;
    expectTopoInfo.version = "2.0";
    expectTopoInfo.peerCount = 3;

    PeerInfo peer0;
    peer0.localId = 0;
    PeerInfo peer1;
    peer1.localId = 1;
    PeerInfo peer2;
    peer2.localId = 2;
    expectTopoInfo.peers.emplace_back(peer0);
    expectTopoInfo.peers.emplace_back(peer1);
    expectTopoInfo.peers.emplace_back(peer2);

    expectTopoInfo.edgeCount = 5;
    EdgeInfo edge0;
    edge0.linkType = LinkType::PEER2PEER;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/1");
    edge0.position = AddrPosition::DEVICE;
    expectTopoInfo.edges.emplace_back(edge0);

    EdgeInfo edge1;
    edge1.linkType = LinkType::PEER2PEER;
    edge1.protocols.emplace(LinkProtocol::UB_MEM);
    edge1.topoType = TopoType::MESH_1D;
    edge1.topoInstId = 0;
    edge1.localA = 0;
    edge1.localAPorts.emplace("0/0");
    edge1.localB = 2;
    edge1.localBPorts.emplace("0/1");
    edge1.position = AddrPosition::DEVICE;
    expectTopoInfo.edges.emplace_back(edge1);

    EdgeInfo edge2;
    edge2.linkType = LinkType::PEER2NET;
    edge2.protocols.emplace(LinkProtocol::UB_CTP);
    edge2.topoType = TopoType::MESH_1D;
    edge2.topoInstId = 0;
    edge2.localA = 0;
    edge2.localAPorts.emplace("0/0");
    edge2.position = AddrPosition::HOST;
    expectTopoInfo.edges.emplace_back(edge2);

    EdgeInfo edge3;
    edge3.linkType = LinkType::PEER2PEER;
    edge3.protocols.emplace(LinkProtocol::UB_TP);
    edge3.topoType = TopoType::MESH_1D;
    edge3.topoInstId = 0;
    edge3.localA = 1;
    edge3.localAPorts.emplace("0/0");
    edge3.localB = 2;
    edge3.localBPorts.emplace("0/1");
    edge3.position = AddrPosition::DEVICE;
    expectTopoInfo.edges.emplace_back(edge3);

    EdgeInfo edge4;
    edge4.linkType = LinkType::PEER2NET;
    edge4.protocols.emplace(LinkProtocol::ROCE);
    edge4.topoType = TopoType::CLOS;
    edge4.topoInstId = 0;
    edge4.localA = 0;
    edge4.localAPorts.emplace("0/0");
    edge4.position = AddrPosition::DEVICE;
    expectTopoInfo.edges.emplace_back(edge4);

    BinaryStream binStream;
    expectTopoInfo.GetBinStream(binStream);
    TopoInfo reBuildTopo(binStream);

    BinaryStream layoutStream;
    expectTopoInfo.GetBinStream(layoutStream);
    std::string binaryVersion;
    u32 binaryPeerCount = 0;
    u32 binaryEdgeCount = 0;
    size_t binaryPeerSize = 0;
    layoutStream >> binaryVersion >> binaryPeerCount >> binaryEdgeCount >> binaryPeerSize;
    EXPECT_EQ(binaryVersion, expectTopoInfo.version);
    EXPECT_EQ(binaryPeerCount, expectTopoInfo.peerCount);
    EXPECT_EQ(binaryEdgeCount, expectTopoInfo.edgeCount);
    EXPECT_EQ(binaryPeerSize, expectTopoInfo.peers.size());
    for (size_t i = 0; i < binaryPeerSize; i++) {
        PeerInfo binaryPeer(layoutStream);
        EXPECT_EQ(binaryPeer.localId, expectTopoInfo.peers[i].localId);
    }
    size_t edgeGroupCount = 0;
    u32 binaryLayer = 0;
    size_t groupEdgeCount = 0;
    layoutStream >> edgeGroupCount >> binaryLayer >> groupEdgeCount;
    EXPECT_EQ(edgeGroupCount, 1);
    EXPECT_EQ(binaryLayer, 0);
    EXPECT_EQ(groupEdgeCount, expectTopoInfo.edges.size());

    EXPECT_EQ(expectTopoInfo.version, reBuildTopo.version);
    EXPECT_EQ(expectTopoInfo.peerCount, reBuildTopo.peerCount);
    EXPECT_EQ(expectTopoInfo.peers.size(), reBuildTopo.peers.size());
    for (u32 i = 0; i < expectTopoInfo.peers.size(); i++) {
        EXPECT_EQ(expectTopoInfo.peers[i].localId, reBuildTopo.peers[i].localId);
    }
    EXPECT_EQ(expectTopoInfo.edgeCount, reBuildTopo.edgeCount);
    EXPECT_EQ(expectTopoInfo.edges.size(), reBuildTopo.edges.size());

    EXPECT_EQ(expectTopoInfo.edges, reBuildTopo.edges);

    EXPECT_EQ(expectTopoInfo.Describe(), reBuildTopo.Describe());
}

TEST_F(TopoParserTest, Ut_BinaryStream_When_ReadLegacyGroups_Expect_FlattenedEdges)
{
    TopoInfo expectTopoInfo;
    expectTopoInfo.version = "2.0";
    expectTopoInfo.peerCount = 2;
    expectTopoInfo.edgeCount = 2;

    PeerInfo peer0;
    peer0.localId = 0;
    PeerInfo peer1;
    peer1.localId = 1;
    expectTopoInfo.peers = {peer0, peer1};

    EdgeInfo edge0;
    edge0.linkType = LinkType::PEER2PEER;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.topoType = TopoType::MESH_1D;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/1");
    edge0.position = AddrPosition::DEVICE;

    EdgeInfo edge1;
    edge1.linkType = LinkType::PEER2NET;
    edge1.protocols.emplace(LinkProtocol::ROCE);
    edge1.topoType = TopoType::CLOS;
    edge1.topoInstId = 1;
    edge1.localA = 1;
    edge1.localAPorts.emplace("1/0");
    edge1.position = AddrPosition::HOST;
    expectTopoInfo.edges = {edge0, edge1};

    BinaryStream legacyStream;
    legacyStream << expectTopoInfo.version << expectTopoInfo.peerCount << expectTopoInfo.edgeCount;
    legacyStream << expectTopoInfo.peers.size();
    for (const auto& peer : expectTopoInfo.peers) {
        peer.GetBinStream(legacyStream);
    }
    const size_t edgeGroupCount = 2;
    const size_t groupEdgeCount = 1;
    legacyStream << edgeGroupCount;
    legacyStream << static_cast<u32>(1) << groupEdgeCount;
    WriteLegacyEdge(legacyStream, 1, edge0);
    legacyStream << static_cast<u32>(7) << groupEdgeCount;
    WriteLegacyEdge(legacyStream, 7, edge1);

    TopoInfo reBuildTopo(legacyStream);
    EXPECT_EQ(reBuildTopo.version, expectTopoInfo.version);
    EXPECT_EQ(reBuildTopo.peerCount, expectTopoInfo.peerCount);
    EXPECT_EQ(reBuildTopo.edgeCount, expectTopoInfo.edgeCount);
    EXPECT_EQ(reBuildTopo.peers.size(), expectTopoInfo.peers.size());
    EXPECT_EQ(reBuildTopo.edges, expectTopoInfo.edges);
}

TEST_F(TopoParserTest, Ut_BinaryStream_When_LegacyGroupsContainSameEdge_Expect_Merged)
{
    TopoInfo topoInfo;
    topoInfo.version = "2.0";
    topoInfo.peerCount = 2;
    topoInfo.edgeCount = 2;

    PeerInfo peer0;
    peer0.localId = 0;
    PeerInfo peer1;
    peer1.localId = 1;
    topoInfo.peers = {peer0, peer1};

    EdgeInfo edge;
    edge.linkType = LinkType::PEER2PEER;
    edge.protocols.emplace(LinkProtocol::UB_CTP);
    edge.topoType = TopoType::MESH_1D;
    edge.localA = 0;
    edge.localAPorts.emplace("0/0");
    edge.localB = 1;
    edge.localBPorts.emplace("0/1");
    edge.position = AddrPosition::DEVICE;

    BinaryStream legacyStream;
    legacyStream << topoInfo.version << topoInfo.peerCount << topoInfo.edgeCount;
    legacyStream << topoInfo.peers.size();
    for (const auto& peer : topoInfo.peers) {
        peer.GetBinStream(legacyStream);
    }

    const size_t edgeGroupCount = 2;
    const size_t groupEdgeCount = 1;
    legacyStream << edgeGroupCount;
    legacyStream << static_cast<u32>(0) << groupEdgeCount;
    WriteLegacyEdge(legacyStream, 0, edge);
    legacyStream << static_cast<u32>(7) << groupEdgeCount;
    WriteLegacyEdge(legacyStream, 7, edge);

    TopoInfo rebuiltTopo(legacyStream);
    EXPECT_EQ(rebuiltTopo.edgeCount, 1);
    ASSERT_EQ(rebuiltTopo.edges.size(), 1);
    EXPECT_EQ(rebuiltTopo.edges[0], edge);
}

TEST_F(TopoParserTest, Ut_DeserializeBinaryStream_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    JsonParser topoParser;
    TopoInfo topoInfo;
    topoParser.ParseString(BuildLargeTopoString(), topoInfo);

    BinaryStream binaryStream;
    topoInfo.GetBinStream(binaryStream);
    TopoInfo reBuildTopoInfo(binaryStream);
    EXPECT_EQ(topoInfo.Describe(), reBuildTopoInfo.Describe());
}
