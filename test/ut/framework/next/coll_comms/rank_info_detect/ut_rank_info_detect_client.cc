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
#include "hccp_peer_manager.h"
#include "timeout_exception.h"
#include "orion_adapter_rts.h"
#include "invalid_params_exception.h"
#include "network_api_exception.h"
#include "host_socket_handle_manager.h"
#include "hcomm_res.h"
#include "whitelist.h"
#include "hccp_peer_manager.h"
#include "dev_type.h"

#include "socket.h"
#include <chrono>
#include <cstddef>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "json_parser.h"
#include "root_handle_v2.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "null_ptr_exception.h"
#include "rank_info_dispatcher.h"
#define private public
#define protected public
#include "rank_info_detect_client.h"
#undef protected
#undef private
#include "orion_adapter_rts.h"
#include "env_config/env_config_v2.h"
#include "base_config_legacy.h"
#include "phy_topo.h"
#include "phy_topo_builder.h"
#include "rank_graph_test_data_builder.h"

extern "C" int TopoAddrInfoGetSize(int phyId, size_t* size);
extern "C" int TopoAddrInfoGet(int phyId, char* rankInfo, size_t* bufSize);
extern "C" int TopoAddrInfoGetTopoFilePath(int phyId, char* filePath, size_t bufSize);

using namespace Hccl;

class RankInfoDetectClientTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankInfoDetectClientTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankInfoDetectClientTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankInfoDetectClientTest SetUP" << std::endl;
        socketHandle = new int(0);
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(socketHandle));
        MOCKER(HrtRaSocketDeInit).stubs().with(mockcpp::any());
        MOCKER_CPP(&HccpPeerManager::Init).stubs().with(mockcpp::any());
        MOCKER_CPP(&HccpPeerManager::DeInit).stubs().with(mockcpp::any());
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER_CPP(&HostSocketHandleManager::Destroy).stubs().with(mockcpp::any(), mockcpp::any());
        IpAddress serverIp = IpAddress("10.0.0.10");
        u32 hostPort = 60001;
        IpAddress hostIp_ = IpAddress("192.168.1.8");
        u32 rankSize_ = 1;
        u32 devPhyId_ = 0;
        u32 rankId_ = 0;
        std::string clientSocketTag = "rank_info_test_server";

        auto clientSocket_ = std::make_shared<Socket>(
            socketHandle, hostIp_, hostPort, serverIp, clientSocketTag, SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
        SocketAgent socketAgent_ = SocketAgent(clientSocket_.get());
        rankInfoDetectClient_ = new RankInfoDetectClient(devPhyId_, rankSize_, rankId_, clientSocket_);
    }

    virtual void TearDown()
    {
        delete rankInfoDetectClient_;
        rankInfoDetectClient_ = nullptr;
        // RankInfoDetectClient 析构会 detach 线程调 DeInit，需等其完成再 verify 清 mock
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        delete static_cast<int*>(socketHandle);
        socketHandle = nullptr;
        GlobalMockObject::verify();
        std::cout << "A Test case in RankInfoDetectClientTest TearDown" << std::endl;
    }

    RankInfoDetectClient* rankInfoDetectClient_{nullptr};
    SocketHandle socketHandle;
};

namespace {
std::string BuildRootInfoJsonString()
{
    nlohmann::json rootInfo;
    rootInfo["version"] = "2.0";
    rootInfo["topo_file_path"] = "/tmp";
    rootInfo["rank_count"] = 2;
    rootInfo["rank_list"] = nlohmann::json::array();
    for (u32 rankId = 0; rankId < 2U; ++rankId) {
        nlohmann::json rank;
        rank["rank_id"] = rankId;
        rank["device_id"] = rankId;
        rank["local_id"] = rankId;
        rank["level_list"] = nlohmann::json::array({
            {
                {"net_layer", 0},
                {"net_instance_id", "az0-rack0"},
                {"net_type", "TOPO_FILE_DESC"},
                {"net_attr", ""},
                {"rank_addr_list", nlohmann::json::array({
                                       {
                                           {"addr_type", "IPV4"},
                                           {"addr", rankId == 0U ? "223.0.0.28" : "223.0.0.10"},
                                           {"ports", nlohmann::json::array({rankId == 0U ? "0/0" : "0/1"})},
                                       },
                                   })},
            },
        });
        rootInfo["rank_list"].push_back(rank);
    }
    return rootInfo.dump();
}

int FillTmpTopoPath(int phyId, char* path, size_t bufSize)
{
    (void)phyId;
    constexpr const char* tmpPath = "/tmp";
    const size_t pathSize = std::strlen(tmpPath) + 1U;
    if (path == nullptr || bufSize < pathSize) {
        return -1;
    }
    std::memcpy(path, tmpPath, pathSize);
    return 0;
}

void BuildEmptyTopo(PhyTopoBuilder* builder, const std::string& topoPath)
{
    (void)builder;
    (void)topoPath;
    PhyTopo::GetInstance()->Clear();
    PhyTopoBuilder::GetInstance().RecoverBuild(test::MakeTopoInfo({0}, {}));
}

void BuildHostRdmaTopo(PhyTopoBuilder* builder, const std::string& topoPath)
{
    (void)builder;
    (void)topoPath;
    PhyTopo::GetInstance()->Clear();
    PhyTopoBuilder::GetInstance().RecoverBuild(test::MakeHostRdmaTopo());
}

void MockTopoPathAndBuild(void (*buildFunc)(PhyTopoBuilder*, const std::string&))
{
    MOCKER(TopoAddrInfoGetTopoFilePath)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(FillTmpTopoPath));
    MOCKER_CPP(&PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(invoke(buildFunc));
}

NewRankInfo BuildRankInfoForTls(u32 rankId, TlsStatus tlsStatus)
{
    NewRankInfo rankInfo{};
    rankInfo.rankId = rankId;
    rankInfo.localId = rankId;
    rankInfo.replacedLocalId = rankId;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    rankInfo.tlsStatus = tlsStatus;
    return rankInfo;
}

void BuildRankTableForTls(RankTableInfo& rankTable, const std::vector<TlsStatus>& tlsStatusList)
{
    rankTable.version = "2.0";
    rankTable.rankCount = tlsStatusList.size();
    rankTable.ranks.clear();
    for (u32 idx = 0; idx < tlsStatusList.size(); ++idx) {
        rankTable.ranks.emplace_back(BuildRankInfoForTls(idx, tlsStatusList[idx]));
    }
}

nlohmann::json BuildLocalDevInfoWithBackupAddrs(
    const std::string& addrType, const std::string& primaryAddr, const nlohmann::json& backupAddrs, u32 netLayer = 3,
    const std::string& port = "host0")
{
    nlohmann::json addrJson = {
        {"addr_type", addrType},
        {"addr", primaryAddr},
        {"backup_addr", backupAddrs},
        {"ports", nlohmann::json::array({port})},
    };
    nlohmann::json levelJson;
    levelJson["net_layer"] = netLayer;
    levelJson["rank_addr_list"] = nlohmann::json::array({addrJson});
    nlohmann::json localDevInfoJson;
    localDevInfoJson["local_id"] = 0;
    localDevInfoJson["level_list"] = nlohmann::json::array({levelJson});
    return localDevInfoJson;
}

nlohmann::json BuildIpv4BackupAddrs(std::size_t count)
{
    nlohmann::json backupAddrs = nlohmann::json::array();
    for (std::size_t idx = 0; idx < count; ++idx) {
        backupAddrs.push_back("192.168.2." + std::to_string(idx + 1));
    }
    return backupAddrs;
}

std::string GetSelectedHostAddr(const nlohmann::json& localDevInfoJson)
{
    return localDevInfoJson["level_list"][0]["rank_addr_list"][0]["addr"].get<std::string>();
}

} // namespace

TEST_F(RankInfoDetectClientTest, Ut_CheckStatus_When_Normal_Expect_Success)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().then(returnValue((SocketStatus)SocketStatus::OK));

    EXPECT_NO_THROW(rankInfoDetectClient_->CheckStatus());
}

TEST_F(RankInfoDetectClientTest, Ut_SendAgentIdAndRankSize_When_Normal_Expect_Success)
{
    MOCKER(HrtRaSocketBlockSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));

    EXPECT_NO_THROW(rankInfoDetectClient_->SendAgentIdAndRankSize());
}

TEST_F(RankInfoDetectClientTest, Ut_ConstructSingleRank_When_Normal_Expect_Success)
{
    RankTableInfo localRankTable;

    EXPECT_NO_THROW(rankInfoDetectClient_->ConstructSingleRank(localRankTable));

    EXPECT_EQ(localRankTable.version, "2.0");
    EXPECT_EQ(localRankTable.rankCount, 1U);
    EXPECT_EQ(localRankTable.ranks.size(), 1U);
    const NewRankInfo& actualRankInfo = localRankTable.ranks[0];
    EXPECT_EQ(actualRankInfo.rankId, 0);
    EXPECT_EQ(actualRankInfo.rankLevelInfos.size(), 1U);
}

TEST_F(RankInfoDetectClientTest, Ut_ConstructRankTable_When_Normal_Expect_Success)
{
    rankInfoDetectClient_->rankSize_ = 2;
    RankTableInfo localRankTable;
    std::string testJsonPath{"/tmp"};

    MOCKER(realpath)
        .stubs()
        .with(mockcpp::any(), outBoundP(const_cast<char*>(testJsonPath.c_str()), testJsonPath.size() + 1))
        .will(returnValue(const_cast<char*>(testJsonPath.c_str())));
    std::string testJsonContent = BuildRootInfoJsonString();
    size_t expectedSize = testJsonContent.size();
    MOCKER(TopoAddrInfoGetSize).stubs().with(0, outBoundP(&expectedSize, sizeof(size_t))).will(returnValue(0));
    const char* rootInfoCtx = testJsonContent.c_str();
    MOCKER(TopoAddrInfoGet)
        .stubs()
        .with(
            0, outBoundP(const_cast<char*>(rootInfoCtx), testJsonContent.size()),
            outBoundP(&expectedSize, sizeof(size_t)))
        .will(returnValue(0));

    EXPECT_NO_THROW(rankInfoDetectClient_->ConstructRankTable(localRankTable));

    EXPECT_EQ(localRankTable.version, "2.0");
    EXPECT_EQ(localRankTable.rankCount, 2);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrAvailable_Expect_SelectBackup)
{
    nlohmann::json localDevInfoJson = nlohmann::json::parse(R"({
        "rank_id": 0,
        "device_id": 0,
        "local_id": 0,
        "level_list": [
            {
                "net_layer": 3,
                "net_instance_id": "cluster",
                "net_type": "CLOS",
                "net_attr": "",
                "rank_addr_list": [
                    {
                        "addr_type": "IPV4",
                        "addr": "192.168.1.101",
                        "backup_addr": ["192.168.1.102", "192.168.1.103"],
                        "ports": ["d2h"],
                        "plane_id": "roce"
                    }
                ]
            }
        ]
    })");

    MOCKER(HcommEndpointCreate)
        .expects(exactly(2))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)))
        .then(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(localDevInfoJson["level_list"][0]["rank_addr_list"][0]["addr"].get<std::string>(), "192.168.1.102");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_LevelListMissing_Expect_Throw)
{
    nlohmann::json localDevInfoJson = {
        {"rank_id", 0},
        {"device_id", 0},
        {"local_id", 0},
    };

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InvalidParamsException);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_LevelListIsNotArray_Expect_Throw)
{
    nlohmann::json localDevInfoJson = {
        {"rank_id", 0},
        {"device_id", 0},
        {"local_id", 0},
        {"level_list", "invalid"},
    };

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InvalidParamsException);
}

TEST_F(
    RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_PrimaryAvailableAndMultipleBackups_Expect_KeepPrimary)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102", "192.168.1.103"}));
    MOCKER(HcommEndpointCreate)
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(
    RankInfoDetectClientTest,
    Ut_SelectLocalHostBackupAddr_When_FirstBackupUnavailableAndSecondAvailable_Expect_SelectSecondBackup)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102", "192.168.1.103"}));
    MOCKER(HcommEndpointCreate)
        .expects(exactly(3))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)))
        .then(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)))
        .then(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.103");
}

TEST_F(
    RankInfoDetectClientTest,
    Ut_SelectLocalHostBackupAddr_When_BackupAddrEmptyAndPrimaryAvailable_Expect_ProbeAndKeepPrimary)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array());
    MOCKER(HcommEndpointCreate)
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Ipv6PrimaryUnavailable_Expect_SelectIpv6Backup)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV6", "2001:db8::1", nlohmann::json::array({"2001:db8::2", "2001:db8::3"}));
    MOCKER(HcommEndpointCreate)
        .expects(exactly(2))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)))
        .then(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "2001:db8::2");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_PrimaryAddrInvalid_Expect_Throw)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1", nlohmann::json::array({"192.168.1.102"}));

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InvalidParamsException);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrInvalid_Expect_Throw)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1"}));

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InvalidParamsException);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrElementNotString_Expect_Throw)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array({102}));

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InvalidParamsException);
}

TEST_F(
    RankInfoDetectClientTest,
    Ut_SelectLocalHostBackupAddr_When_PrimaryProbeParameterError_Expect_ThrowInternalException)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102", "192.168.1.103"}));
    MOCKER(HcommEndpointCreate)
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_PARA)));

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InternalException);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_NoBackupAddr_Expect_KeepPrimaryWithoutProbe)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array());
    localDevInfoJson["level_list"][0]["rank_addr_list"][0].erase("backup_addr");
    MOCKER(HcommEndpointCreate).expects(never());

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Layer4HasBackupAddr_Expect_SelectBackup)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102"}), 4);
    MOCKER(HcommEndpointCreate)
        .expects(exactly(2))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)))
        .then(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.102");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Layer0To2HaveBackupAddr_Expect_KeepPrimary)
{
    MOCKER(HcommEndpointCreate).expects(never());
    for (u32 netLayer = 0; netLayer <= 2; ++netLayer) {
        nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
            "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102"}), netLayer);

        EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
        EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.101");
    }
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Layer3PortsAreUnrelated_Expect_SelectBackup)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", nlohmann::json::array({"192.168.1.102"}), 3, "unrelated-port");
    MOCKER(HcommEndpointCreate)
        .expects(exactly(2))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)))
        .then(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(GetSelectedHostAddr(localDevInfoJson), "192.168.1.102");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_Layer3HasMultipleAddrConfigs_Expect_SelectEachBackup)
{
    nlohmann::json localDevInfoJson = nlohmann::json::parse(R"({
        "local_id": 0,
        "level_list": [{
            "net_layer": 3,
            "rank_addr_list": [
                {
                    "addr_type": "IPV4",
                    "addr": "192.168.1.10",
                    "backup_addr": ["192.168.1.11"],
                    "ports": ["port-a"]
                },
                {
                    "addr_type": "IPV4",
                    "addr": "192.168.2.10",
                    "backup_addr": ["192.168.2.11"],
                    "ports": ["port-b"]
                }
            ]
        }]
    })");
    MOCKER(HcommEndpointCreate)
        .expects(exactly(4))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)))
        .then(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)))
        .then(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)))
        .then(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    EXPECT_NO_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson));
    EXPECT_EQ(localDevInfoJson["level_list"][0]["rank_addr_list"][0]["addr"].get<std::string>(), "192.168.1.11");
    EXPECT_EQ(localDevInfoJson["level_list"][0]["rank_addr_list"][1]["addr"].get<std::string>(), "192.168.2.11");
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_ProbeReturnsNotSupport_Expect_ThrowInternalException)
{
    nlohmann::json localDevInfoJson = nlohmann::json::parse(R"({
        "rank_id": 0,
        "device_id": 0,
        "local_id": 0,
        "level_list": [
            {
                "net_layer": 3,
                "net_instance_id": "cluster",
                "net_type": "CLOS",
                "net_attr": "",
                "rank_addr_list": [
                    {
                        "addr_type": "IPV4",
                        "addr": "192.168.1.101",
                        "backup_addr": ["192.168.1.102"],
                        "ports": ["d2h"],
                        "plane_id": "roce"
                    }
                ]
            }
        ]
    })");

    MOCKER(HcommEndpointCreate)
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NOT_SUPPORT)));

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InternalException);
    EXPECT_EQ(localDevInfoJson["level_list"][0]["rank_addr_list"][0]["addr"].get<std::string>(), "192.168.1.101");
}

TEST_F(
    RankInfoDetectClientTest,
    Ut_SelectLocalHostBackupAddr_When_AllCandidatesReturnNetworkError_Expect_ThrowNetworkApiException)
{
    nlohmann::json localDevInfoJson = nlohmann::json::parse(R"({
        "local_id": 0,
        "level_list": [
            {
                "net_layer": 3,
                "rank_addr_list": [
                    {
                        "addr_type": "IPV4",
                        "addr": "192.168.1.101",
                        "backup_addr": ["192.168.1.102", "192.168.1.103"],
                        "ports": ["d2h"]
                    }
                ]
            }
        ]
    })");

    MOCKER(HcommEndpointCreate)
        .expects(exactly(3))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)));

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), NetworkApiException);
}

TEST_F(
    RankInfoDetectClientTest,
    Ut_SelectLocalHostBackupAddr_When_BackupAddrCountAtLimitAndAllUnavailable_Expect_ThrowNetworkApiException)
{
    nlohmann::json localDevInfoJson
        = BuildLocalDevInfoWithBackupAddrs("IPV4", "192.168.1.101", BuildIpv4BackupAddrs(MAX_VALUE_BACKUP_ADDR_SIZE));
    MOCKER(HcommEndpointCreate)
        .expects(exactly(MAX_VALUE_BACKUP_ADDR_SIZE + 1U))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)));

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), NetworkApiException);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrCountExceedsLimit_Expect_ThrowBeforeProbe)
{
    nlohmann::json localDevInfoJson = BuildLocalDevInfoWithBackupAddrs(
        "IPV4", "192.168.1.101", BuildIpv4BackupAddrs(MAX_VALUE_BACKUP_ADDR_SIZE + 1U));
    MOCKER(HcommEndpointCreate).expects(never());

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InvalidParamsException);
}

TEST_F(RankInfoDetectClientTest, Ut_SelectLocalHostBackupAddr_When_BackupAddrIsNotArray_Expect_Throw)
{
    nlohmann::json localDevInfoJson = nlohmann::json::parse(R"({
        "local_id": 0,
        "level_list": [
            {
                "net_layer": 3,
                "rank_addr_list": [
                    {
                        "addr_type": "IPV4",
                        "addr": "192.168.1.101",
                        "backup_addr": "192.168.1.102",
                        "ports": ["d2h"]
                    }
                ]
            }
        ]
    })");

    EXPECT_THROW(rankInfoDetectClient_->SelectLocalHostBackupAddr(localDevInfoJson), InvalidParamsException);
}

TEST_F(RankInfoDetectClientTest, Ut_ProbeHostRoceAddr_When_EndpointCreated_Expect_DestroyAndNoThrow)
{
    int endpoint = 0;
    EndpointHandle endpointHandle = static_cast<EndpointHandle>(&endpoint);
    MOCKER(HcommEndpointCreate)
        .expects(exactly(1))
        .with(mockcpp::any(), outBoundP(&endpointHandle))
        .will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(HcommEndpointDestroy)
        .expects(exactly(1))
        .with(endpointHandle)
        .will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    bool isAvailable = false;
    EXPECT_NO_THROW(rankInfoDetectClient_->ProbeHostRoceAddr(IpAddress("192.168.1.101", AF_INET), isAvailable));
    EXPECT_TRUE(isAvailable);
}

TEST_F(RankInfoDetectClientTest, Ut_ProbeHostRoceAddr_When_EndpointCreateReturnsNetworkError_Expect_MarkUnavailable)
{
    MOCKER(HcommEndpointCreate)
        .expects(exactly(1))
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_NETWORK)));

    bool isAvailable = true;
    EXPECT_NO_THROW(rankInfoDetectClient_->ProbeHostRoceAddr(IpAddress("192.168.1.101", AF_INET), isAvailable));
    EXPECT_FALSE(isAvailable);
}

TEST_F(RankInfoDetectClientTest, Ut_ProbeHostRoceAddr_When_Ipv6EndpointCreated_Expect_NoThrow)
{
    int endpoint = 0;
    EndpointHandle endpointHandle = static_cast<EndpointHandle>(&endpoint);
    MOCKER(HcommEndpointCreate)
        .expects(exactly(1))
        .with(mockcpp::any(), outBoundP(&endpointHandle))
        .will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(HcommEndpointDestroy)
        .expects(exactly(1))
        .with(endpointHandle)
        .will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    bool isAvailable = false;
    EXPECT_NO_THROW(rankInfoDetectClient_->ProbeHostRoceAddr(IpAddress("2001:db8::1", AF_INET6), isAvailable));
    EXPECT_TRUE(isAvailable);
}

TEST_F(RankInfoDetectClientTest, Ut_ProbeHostRoceAddr_When_EndpointDestroyFails_Expect_ThrowInternal)
{
    int endpoint = 0;
    EndpointHandle endpointHandle = static_cast<EndpointHandle>(&endpoint);
    MOCKER(HcommEndpointCreate)
        .expects(exactly(1))
        .with(mockcpp::any(), outBoundP(&endpointHandle))
        .will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(HcommEndpointDestroy)
        .expects(exactly(1))
        .with(endpointHandle)
        .will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));

    bool isAvailable = false;
    EXPECT_THROW(
        rankInfoDetectClient_->ProbeHostRoceAddr(IpAddress("192.168.1.101", AF_INET), isAvailable), InternalException);
}

TEST_F(RankInfoDetectClientTest, Ut_RecvRankTable_When_Normal_Expect_Success)
{
    RankTableInfo localRankTable;
    localRankTable.version = "1.0";
    localRankTable.rankCount = 1;
    NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);

    BinaryStream binaryStream;
    localRankTable.GetBinStream(true, binaryStream);
    binaryStream << rankInfoDetectClient_->currentStep_;
    std::string temp = "";
    binaryStream << temp;

    // 字节流转换为vector<char>格式
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);

    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));
    std::vector<char> hostAlloc(MAX_BUFFER_LEN);
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(hostAlloc.data())));
    MOCKER(HrtFreeHost).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    void* msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64 revMsgLenOut = msgLen;
    u64& revMsgLen = revMsgLenOut;
    MOCKER_CPP(&SocketAgent::RecvMsg).stubs().with(outBoundP(msg, msgLen), outBound(revMsgLen)).will(returnValue(true));

    MOCKER_CPP(&RankInfoDetectClient::VerifyRankTable).stubs().will(ignoreReturnValue());

    EXPECT_NO_THROW(rankInfoDetectClient_->RecvRankTable());
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_AllRanksEnable_Expect_ReturnSuccess)
{
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::ENABLE});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_AllRanksDisable_Expect_ReturnSuccess)
{
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::DISABLE, TlsStatus::DISABLE});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_EnableAndDisableMixed_Expect_ReturnParaError)
{
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::DISABLE});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_KnownConsistentAndUnknownExists_Expect_ReturnSuccess)
{
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::UNKNOWN});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankInfoDetectClientTest, Ut_VerifyTlsConsistency_When_KnownInconsistentAndUnknownExists_Expect_ReturnParaError)
{
    BuildRankTableForTls(
        rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::DISABLE, TlsStatus::UNKNOWN});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_CheckStatus_When_Timeout_Expect_Throw)
{
    EnvSocketConfig fakeEnvSocketConfig;
    fakeEnvSocketConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(1), Str2T<s32>};
    fakeEnvSocketConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvSocketConfig));
    MOCKER_CPP(&Socket::GetStatus).stubs().then(returnValue((SocketStatus)SocketStatus::CONNECTING));

    EXPECT_THROW(rankInfoDetectClient_->CheckStatus(), TimeoutException);
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_EmptyRankLevelInfos_Expect_NoThrow)
{
    // Given: rankInfo with empty rankLevelInfos
    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;
    // rankLevelInfos is empty by default

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    MockTopoPathAndBuild(BuildEmptyTopo);

    // When & Then: loop iterates zero times, returns without error
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_NoTopoGraph_Expect_NoThrow)
{
    // Given: rankInfo with rankLevelInfos but netLayer doesn't match any built topo graph
    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    AddressInfo addrInfo;
    addrInfo.addr = IpAddress("192.168.1.1");
    addrInfo.socketPort_ = 0;
    RankLevelInfo levelInfo;
    levelInfo.netLayer = 0;
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    MockTopoPathAndBuild(BuildEmptyTopo);

    // When: PhyTopo built but netLayer=0 has no graph → GetTopoGraph returns nullptr
    // Then: skip nullptr graph, log debug, continue, return without error
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_MultipleRankLevelInfos_Expect_NoThrow)
{
    // Given: multiple rankLevelInfos, none with a matching topo graph
    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    for (u32 i = 0; i < 3; i++) {
        AddressInfo addrInfo;
        addrInfo.addr = IpAddress(StringFormat("192.168.%u.1", i + 1));
        addrInfo.socketPort_ = 0;
        RankLevelInfo levelInfo;
        levelInfo.netLayer = i;
        levelInfo.rankAddrs.push_back(addrInfo);
        rankInfo.rankLevelInfos.push_back(levelInfo);
    }

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    MockTopoPathAndBuild(BuildEmptyTopo);

    // When & Then: iterate all level infos, all graphs nullptr, no throw
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_RdmaLinkEmptyRankAddrs_Expect_Continue)
{
    // Given: host RDMA topology exists at netLayer=3, but rankAddrs is empty
    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    RankLevelInfo levelInfo;
    levelInfo.netLayer = 3; // HOST+ROCE link at netLayer=3
    // rankAddrs left EMPTY — triggers rankLevelInfo.rankAddrs.empty() check
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    MockTopoPathAndBuild(BuildHostRdmaTopo);

    // When & Then: ROCE → RDMA but rankAddrs.empty() → skip, hostPort unchanged
    PhyTopo::GetInstance()->Clear();
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

// basePort configured, portRange empty → listenPort = basePort + devPhyId
TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_BasePort_Expect_HostPortSet)
{
    // Given: HOST+ROCE link at netLayer=3, basePort configured
    EnvHostNicConfig fakeConfig;
    fakeConfig.hcclHostSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_HOST_SOCKET_PORT_RANGE", {}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_HOST_SOCKET_PORT_RANGE");
        }};
    fakeConfig.hcclHostSocketPortRange.isParsed = true;
    // hcclIfBasePort: configured to valid value 50000
    fakeConfig.hcclIfBasePort = CfgField<u32>{"HCCL_IF_BASE_PORT", 50000, Str2T<u32>};
    fakeConfig.hcclIfBasePort.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeConfig));

    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    AddressInfo addrInfo;
    addrInfo.addr = IpAddress("192.168.1.1");
    RankLevelInfo levelInfo;
    levelInfo.netLayer = 3; // HOST+ROCE link at netLayer=3
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    MockTopoPathAndBuild(BuildHostRdmaTopo);

    // When: basePort valid → listenPort = basePort + devPhyId = 50000 + 0
    PhyTopo::GetInstance()->Clear();
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));

    // Then: hostPort should be set to basePort + devPhyId
    EXPECT_EQ(rankInfo.hostPort, 50000u);

    // Cleanup: TearDown resets hostSocket_
    rankInfoDetectClient_->SocketTearDown(0);
    PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_NoHostLink_Expect_HostPortUnchanged)
{
    // Given: graph exists at netLayer=1, but links are DEVICE position instead of HOST
    // This tests the path where graph has edges but none match HOST position
    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    AddressInfo addrInfo;
    addrInfo.addr = IpAddress("192.168.1.1");
    RankLevelInfo levelInfo;
    levelInfo.netLayer = 1; // netLayer=1 has DEV links, not HOST
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    MockTopoPathAndBuild(BuildHostRdmaTopo);

    // When: netLayer=1 graph exists but has no HOST position links
    PhyTopo::GetInstance()->Clear();
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));

    // Then: hostPort should remain at default since no HOST RDMA link found
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear(); // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_TearDown_HostSocketNull_Expect_EarlyReturn)
{
    // Given: hostSocket_ is nullptr (default state, no HostListenPortDetect called)

    // When & Then: TearDown should return immediately without any side effects
    EXPECT_NO_THROW(rankInfoDetectClient_->SocketTearDown(0));
}
