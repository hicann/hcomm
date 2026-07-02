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
#include "host_socket_handle_manager.h"
#include "whitelist.h"
#include "hccp_peer_manager.h"
#include "dev_type.h"

#include "socket.h"
#include <chrono>
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
#include "env_config.h"
#include "base_config.h"
#include "topo_addr_info.h"
#include "communicator_impl.h"

using namespace Hccl;

std::string filePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/topo/rank_info_detect/rootinfo.json"};

class RankInfoDetectClientTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "RankInfoDetectClientTest SetUP" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "RankInfoDetectClientTest TearDown" << std::endl;
    }

    virtual void SetUp() {
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
            socketHandle, hostIp_, hostPort, serverIp,
            clientSocketTag, SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE
        );
        SocketAgent socketAgent_ = SocketAgent(clientSocket_.get());
        rankInfoDetectClient_ = new RankInfoDetectClient(devPhyId_, rankSize_, rankId_, clientSocket_);
    }

    virtual void TearDown() {
        delete rankInfoDetectClient_;
        rankInfoDetectClient_ = nullptr;
        // RankInfoDetectClient 析构会 detach 线程调 DeInit，需等其完成再 verify 清 mock
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        delete static_cast<int *>(socketHandle);
        socketHandle = nullptr;
        GlobalMockObject::verify();
        std::cout << "A Test case in RankInfoDetectClientTest TearDown" << std::endl;
    }

    RankInfoDetectClient *rankInfoDetectClient_{nullptr};
    nlohmann::json presetParseJson_;       // 模拟ParseFileToJson的返回结果
    nlohmann::json presetLocalDevJson_;    // 模拟GetLocalDevInfoJson的返回结果
    nlohmann::json presetRankTableJson_;   // 模拟GetLocalRankTableJson的返回结果
    SocketHandle socketHandle;
};

namespace {
NewRankInfo BuildRankInfoForTls(u32 rankId, TlsStatus tlsStatus)
{
    NewRankInfo rankInfo {};
    rankInfo.rankId = rankId;
    rankInfo.localId = rankId;
    rankInfo.replacedLocalId = rankId;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo {});
    rankInfo.tlsStatus = tlsStatus;
    return rankInfo;
}

void BuildRankTableForTls(RankTableInfo &rankTable, const std::vector<TlsStatus> &tlsStatusList)
{
    rankTable.version = "2.0";
    rankTable.rankCount = tlsStatusList.size();
    rankTable.ranks.clear();
    for (u32 idx = 0; idx < tlsStatusList.size(); ++idx) {
        rankTable.ranks.emplace_back(BuildRankInfoForTls(idx, tlsStatusList[idx]));
    }
}
}

TEST_F(RankInfoDetectClientTest, Ut_CheckStatus_When_Normal_Expect_Success)
{
    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::OK));

    EXPECT_NO_THROW(rankInfoDetectClient_->CheckStatus());
}

TEST_F(RankInfoDetectClientTest, Ut_SendAgentIdAndRankSize_When_Normal_Expect_Success)
{
    MOCKER(HrtRaSocketBlockSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(true));

    EXPECT_NO_THROW(rankInfoDetectClient_->SendAgentIdAndRankSize());
}

TEST_F(RankInfoDetectClientTest, Ut_ConstructSingleRank_When_Normal_Expect_Success)
{
    RankTableInfo localRankTable;

    EXPECT_NO_THROW(rankInfoDetectClient_->ConstructSingleRank(localRankTable));

    EXPECT_EQ(localRankTable.version, "2.0");
    EXPECT_EQ(localRankTable.rankCount, 1U) ;
    EXPECT_EQ(localRankTable.ranks.size(), 1U);
    const NewRankInfo& actualRankInfo = localRankTable.ranks[0];
    EXPECT_EQ(actualRankInfo.rankId, 0); 
    EXPECT_EQ(actualRankInfo.rankLevelInfos.size(), 1U); 
}

TEST_F(RankInfoDetectClientTest, Ut_ConstructRankTable_When_Normal_Expect_Success)
{
    rankInfoDetectClient_->rankSize_ = 2;
    RankTableInfo localRankTable;
    std::string testJsonPath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/topo/rank_info_detect/rootInfo.json"};

    MOCKER(realpath) 
        .stubs()  
        .with(
            mockcpp::any(), 
            outBoundP(
                const_cast<char*>(testJsonPath.c_str()),  
                testJsonPath.size() + 1                   
            )
        )
        .will(returnValue(
            const_cast<char*>(testJsonPath.c_str()) 
        ));
    std::string testJsonContent = R"({
        "version": "2.0",
        "topo_file_path": "rootInfo.json",
        "rank_count" : 2,
        "rank_list": [
            {
                "rank_id": 0,
                "device_id": 0,
                "local_id": 0,
                "level_list":  [
                    {
                        "net_layer": 0,
                        "net_instance_id" : "az0-rack0",
                        "net_type": "TOPO_FILE_DESC",
                        "net_attr": "",
                        "rank_addr_list": [
                        {
                        "addr_type": "IPV4",
                        "addr": "223.0.0.28",
                        "ports": [ "0/0" ]
                        }
                        ]
                    }
                ]
            },
            {
                "rank_id": 1,
                "local_id": 1,
                "device_id": 1,
                "level_list":  [
                    {
                        "net_layer": 0,
                        "net_instance_id" : "az0-rack0",
                        "net_type": "TOPO_FILE_DESC",
                        "net_attr": "",
                        "rank_addr_list": [
                        {
                        "addr_type": "IPV4",
                        "addr": "223.0.0.10",
                        "ports": [ "0/1" ]
                        }
                        ]
                    }
                ]
            }
        ]
    })";
    size_t expectedSize = testJsonContent.size();
    MOCKER(TopoAddrInfoGetSize)
        .stubs()
        .with(0, outBoundP(&expectedSize, sizeof(size_t)))
        .will(returnValue(0));
    const char* rootInfoCtx = testJsonContent.c_str();
    MOCKER(TopoAddrInfoGet)
        .stubs()
        .with(0, outBoundP(const_cast<char*>(rootInfoCtx), testJsonContent.size()), outBoundP(&expectedSize, sizeof(size_t)))
        .will(returnValue(0));

    EXPECT_NO_THROW(rankInfoDetectClient_->ConstructRankTable(localRankTable));

    EXPECT_EQ(localRankTable.version, "2.0");
    EXPECT_EQ(localRankTable.rankCount, 2);

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
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void *>(hostAlloc.data())));
    MOCKER(HrtFreeHost).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    void *msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64 revMsgLenOut = msgLen;
    u64 &revMsgLen = revMsgLenOut;
    MOCKER_CPP(&SocketAgent::RecvMsg)
            .stubs()
            .with(outBoundP(msg, msgLen), outBound(revMsgLen))
            .will(returnValue(true));

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
    BuildRankTableForTls(rankInfoDetectClient_->rankTable_, {TlsStatus::ENABLE, TlsStatus::DISABLE, TlsStatus::UNKNOWN});

    HcclResult ret = rankInfoDetectClient_->VerifyTlsConsistency();

    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankInfoDetectClientTest, Ut_CheckStatus_When_Timeout_Expect_Throw)
{
    EnvSocketConfig fakeEnvSocketConfig;
    fakeEnvSocketConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(1), Str2T<s32>};
    fakeEnvSocketConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvSocketConfig));
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::CONNECTING));

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

    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topohost.json"};
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath)
        .stubs()
        .will(returnValue(topoFilePath));

    // When & Then: loop iterates zero times, returns without error
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear();  // 清理上一次测试的拓扑状态
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

    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topohost.json"};
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath)
        .stubs()
        .will(returnValue(topoFilePath));

    // When: PhyTopo built but netLayer=0 has no graph → GetTopoGraph returns nullptr
    // Then: skip nullptr graph, log debug, continue, return without error
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear();  // 清理上一次测试的拓扑状态
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

    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topohost.json"};
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath)
        .stubs()
        .will(returnValue(topoFilePath));

    // When & Then: iterate all level infos, all graphs nullptr, no throw
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear();  // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_RdmaLinkEmptyRankAddrs_Expect_Continue)
{
    // Given: topohost.json has HOST+ROCE at netLayer=3, but rankAddrs is empty
    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    RankLevelInfo levelInfo;
    levelInfo.netLayer = 3;  // HOST+ROCE link at netLayer=3
    // rankAddrs left EMPTY — triggers rankLevelInfo.rankAddrs.empty() check
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topohost.json"};
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath)
        .stubs()
        .will(returnValue(topoFilePath));

    // When & Then: ROCE → RDMA but rankAddrs.empty() → skip, hostPort unchanged
    PhyTopo::GetInstance()->Clear();
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear();  // 清理上一次测试的拓扑状态
}

// basePort configured, portRange empty → listenPort = basePort + devPhyId
TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_BasePort_Expect_HostPortSet)
{
    // Given: HOST+ROCE link at netLayer=3, basePort configured
    EnvHostNicConfig fakeConfig;
    fakeConfig.hcclHostSocketPortRange = CfgField<std::vector<SocketPortRange>>{"HCCL_HOST_SOCKET_PORT_RANGE", {},
        [] (const std::string &s) -> std::vector<SocketPortRange> { return CastSocketPortRange(s, "HCCL_HOST_SOCKET_PORT_RANGE"); }};
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
    levelInfo.netLayer = 3;  // HOST+ROCE link at netLayer=3
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topohost.json"};
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath)
        .stubs()
        .will(returnValue(topoFilePath));

    // When: basePort valid → listenPort = basePort + devPhyId = 50000 + 0
    PhyTopo::GetInstance()->Clear();
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));

    // Then: hostPort should be set to basePort + devPhyId
    EXPECT_EQ(rankInfo.hostPort, 50000u);

    // Cleanup: TearDown resets hostSocket_
    rankInfoDetectClient_->SocketTearDown(0);
    PhyTopo::GetInstance()->Clear();  // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_HostListenPortDetect_NoHostLink_Expect_HostPortUnchanged)
{
    // Given: topohost.json has HOST+ROCE at netLayer=3, but rankAddrs has DEV position instead
    // This tests the path where graph has edges but none match HOST position
    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;

    AddressInfo addrInfo;
    addrInfo.addr = IpAddress("192.168.1.1");
    RankLevelInfo levelInfo;
    levelInfo.netLayer = 1;  // netLayer=1 has DEV links, not HOST
    levelInfo.rankAddrs.push_back(addrInfo);
    rankInfo.rankLevelInfos.push_back(levelInfo);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topohost.json"};
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath)
        .stubs()
        .will(returnValue(topoFilePath));

    // When: netLayer=1 graph exists but has no HOST position links
    PhyTopo::GetInstance()->Clear();
    EXPECT_NO_THROW(rankInfoDetectClient_->HostListenPortDetect(rankInfo));

    // Then: hostPort should remain at default since no HOST RDMA link found
    EXPECT_EQ(rankInfo.hostPort, DEFAULT_VALUE_TCPPORT);
    PhyTopo::GetInstance()->Clear();  // 清理上一次测试的拓扑状态
}

TEST_F(RankInfoDetectClientTest, Ut_TearDown_HostSocketNull_Expect_EarlyReturn)
{
    // Given: hostSocket_ is nullptr (default state, no HostListenPortDetect called)

    // When & Then: TearDown should return immediately without any side effects
    EXPECT_NO_THROW(rankInfoDetectClient_->SocketTearDown(0));
}
