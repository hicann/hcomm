/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include "rank_info_detect_client.h"
#include "root_handle_v2.h"
#include "env_config/env_config_v2.h"
#include "host_buffer.h"
#include "binary_stream.h"
#include "hccp_peer_manager.h"
#include "hcomm_res.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "host_socket_handle_manager.h"
#include "socket_manager.h"
#include "topo_addr_info.h"
#include "adapter_error_manager_pub.h"
#include "network_api_exception.h"
#include "phy_topo_builder.h"
#include "preempt_port_manager_v2.h"

namespace Hccl {
namespace {
    constexpr u32 HOST_CONTROL_PORT_COUNT = 15;
    constexpr u32 HOST_BACKUP_ADDR_NET_LAYER = 3;
    constexpr const char* BACKUP_ADDR_FIELD = "backup_addr";
    constexpr const char* ADDR_FIELD = "addr";
    constexpr const char* ADDR_TYPE_FIELD = "addr_type";

    void FillCommAddr(CommAddr& commAddr, const IpAddress& ipAddr)
    {
        const s32 family = ipAddr.GetFamily();
        if (family == AF_INET) {
            commAddr.type = COMM_ADDR_TYPE_IP_V4;
            commAddr.addr = ipAddr.GetBinaryAddress().addr;
        } else if (family == AF_INET6) {
            commAddr.type = COMM_ADDR_TYPE_IP_V6;
            commAddr.addr6 = ipAddr.GetBinaryAddress().addr6;
        } else {
            THROW<InvalidParamsException>(
                StringFormat("[%s] invalid commAddrType, hostAddr[%s].", __func__, ipAddr.Describe().c_str()));
        }
    }

    void BuildHostAddrCandidates(const nlohmann::json& addrJson, std::vector<IpAddress>& candidates)
    {
        // 候选顺序固定为主地址在前、备地址按配置顺序在后，选择时取首个探测成功的地址。
        std::string addrType;
        std::string primaryAddr;
        const std::string msgAddrType = "get host addr_type failed";
        TRY_CATCH_THROW(InvalidParamsException, msgAddrType, addrType = GetJsonProperty(addrJson, ADDR_TYPE_FIELD););
        const std::string msgPrimaryAddr = "get primary host addr failed";
        TRY_CATCH_THROW(InvalidParamsException, msgPrimaryAddr, primaryAddr = GetJsonProperty(addrJson, ADDR_FIELD););
        IpAddress primaryIpAddress;
        const std::string msgParsePrimaryAddr = "parse primary host addr failed";
        TRY_CATCH_THROW(InvalidParamsException, msgParsePrimaryAddr,
                        AddressInfo::ParseAddrByType(addrType, primaryAddr, primaryIpAddress););
        candidates.clear();
        candidates.emplace_back(primaryIpAddress);

        std::vector<IpAddress> backupAddrs;
        const std::string msgParseBackupAddr = "parse backup host addr failed";
        TRY_CATCH_THROW(InvalidParamsException, msgParseBackupAddr,
                        AddressInfo::ParseBackupAddrs(addrJson.at(BACKUP_ADDR_FIELD), addrType, backupAddrs););
        candidates.insert(candidates.end(), backupAddrs.begin(), backupAddrs.end());
    }

    void CollectLayer3AddrJsons(nlohmann::json& localDevInfoJson, std::vector<nlohmann::json*>& addrJsons)
    {
        // 仅返回 netLayer3 及以上且配置了 backup_addr 的可写地址节点，其他地址直接沿用主 addr。
        addrJsons.clear();
        CHK_PRT_THROW(
            !localDevInfoJson.contains("level_list") || !localDevInfoJson.at("level_list").is_array(),
            HCCL_ERROR("[%s] level_list is missing or is not an array.", __func__), InvalidParamsException,
            "level_list is missing or is not an array");
        for (auto& levelJson : localDevInfoJson.at("level_list")) {
            if (!levelJson.contains("rank_addr_list") || !levelJson["rank_addr_list"].is_array()) {
                continue;
            }
            u32 netLayer = 0;
            const std::string msgNetLayer = "get net_layer failed";
            TRY_CATCH_THROW(InvalidParamsException, msgNetLayer,
                            netLayer = GetJsonPropertyUInt(levelJson, "net_layer"););
            if (netLayer < HOST_BACKUP_ADDR_NET_LAYER) {
                continue;
            }
            for (auto& addrJson : levelJson["rank_addr_list"]) {
                if (!addrJson.contains(BACKUP_ADDR_FIELD)) {
                    HCCL_WARNING("[%s] backup_addr is not configured, use primary addr without probing.", __func__);
                    continue;
                }
                addrJsons.push_back(&addrJson);
            }
        }
    }

    std::string QueryTopoFilePathByDevice()
    {
        const size_t bufSize = 1024;
        auto devLogicId = HrtGetDevice();
        auto devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);
        std::vector<char> buffer(bufSize, '\0');
        int result = TopoAddrInfoGetTopoFilePath(devPhyId, buffer.data(), buffer.size());
        CHK_PRT_THROW(
            result != 0, HCCL_ERROR("[%s] Get topo file path failed.", __func__), InvalidParamsException,
            "Get topo file path failed.");
        return std::string(buffer.data());
    }

    void CheckTopoFilePath(const std::string& topoFilePath)
    {
        char resolvedPath[PATH_MAX] = {0};
        CHK_PRT_THROW(
            realpath(topoFilePath.c_str(), resolvedPath) == nullptr,
            HCCL_ERROR("[%s] topo_file_path[%s] is not a valid real path", __func__, topoFilePath.c_str()),
            InvalidParamsException, "topo_file_path error");
    }

    std::string GetRootInfoTopoFilePath()
    {
        std::string filePath = "/etc/hccl_rootinfo.json";
        JsonParser jsonParser{};
        nlohmann::json parseJson{};
        std::string topoFilePath{};
        std::ifstream file(filePath);
        if (file.good()) {
            jsonParser.ParseFileToJson(filePath, parseJson);
            std::string msgRankTopoFile = "error occurs when parser object of propName \"topo_file_path\"";
            TRY_CATCH_THROW(InvalidParamsException, msgRankTopoFile,
                            topoFilePath = GetJsonProperty(parseJson, "topo_file_path"););
        } else {
            topoFilePath = QueryTopoFilePathByDevice();
        }

        CheckTopoFilePath(topoFilePath);
        return topoFilePath;
    }

    bool IsHostRdmaLink(const std::shared_ptr<PhyTopo::Link>& link)
    {
        if (link == nullptr || link->GetSourceIFace() == nullptr
            || link->GetSourceIFace()->GetPos() != AddrPosition::HOST) {
            return false;
        }
        const std::set<LinkProtocol>& protocols = link->GetLinkProtocols();
        return std::any_of(protocols.begin(), protocols.end(), [](const LinkProtocol& protocol) {
            return LinkProtocol2LinkProtoType(protocol) == LinkProtoType::RDMA;
        });
    }

    bool HasMatchingPort(const AddressInfo& addrInfo, const std::set<std::string>& phyPorts)
    {
        return std::any_of(addrInfo.ports.begin(), addrInfo.ports.end(), [&phyPorts](const std::string& port) {
            return phyPorts.count(port) != 0;
        });
    }
} // namespace

void RankInfoDetectClient::Setup(RankTableInfo& rankTable)
{
    // 1. 构造localRankTable
    RankTableInfo localRankTable{};
    ConstructRankTable(localRankTable);

    // 若启用单卡多进程抢占端口则执行
    SocketManager::ServerInitAll(localRankTable.ranks[0]);
    HostListenPortDetect(localRankTable.ranks[0]);

    // 2. 连接root节点
    Connect();

    // 3. 发送本端agentId和rankSize
    SendAgentIdAndRankSize();

    // 4. 发送给root节点
    SendLocalRankTable(localRankTable);

    // 5. 接收完整rankTable
    RecvRankTable();
    rankTable = rankTable_;
}

void RankInfoDetectClient::Connect()
{
    clientSocket_->Connect();
    CheckStatus();
}

void RankInfoDetectClient::CheckStatus()
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    while (true) {
        bool isTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
        if (isTimeout) {
            HCCL_ERROR(
                "[RankInfoDetectClient::%s] get connected status socket timeout! timeout[%lld s]", __func__, timeout);
            RPT_INPUT_ERR(
                isTimeout, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>({StringFormat(
                    "Receiving message from the root node timed out "
                    "Timeout was set to %lld seconds. Check whether node rankId[%u] reports an error.",
                    static_cast<long long>(timeout.count()), rankId_)}));
            // 建链超时后，sleep 20s，避免上层应用提前退出，确保其他正常 client 能够收到 server 发出的临终遗言
            sleep(WAIT_ERROR_BROADCAST_TIME);
            THROW<TimeoutException>("client get connection timeout");
        }

        if (clientSocket_->GetStatus() == SocketStatus::OK) {
            HCCL_DEBUG("[RankInfoDetectClient::%s] client get socket connection success.", __func__);
            break;
        }
    }

    HCCL_INFO("[RankInfoDetectClient::%s] end, connect ok.", __func__);
}

void RankInfoDetectClient::SendAgentIdAndRankSize()
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    // 发送agentId
    std::string rankID = std::to_string(rankId_);
    std::string agentID = std::string(16 - rankID.length(), '0') + rankID;
    socketAgent_.SendMsg(agentID.c_str(), agentID.size());

    // 发送rankSize
    socketAgent_.SendMsg(&rankSize_, sizeof(rankSize_));

    HCCL_INFO(
        "[RankInfoDetectClient::%s] send agentID[%s] and rankSize_[%u] end.", __func__, agentID.c_str(), rankSize_);
}

void RankInfoDetectClient::SendLocalRankTable(const RankTableInfo& localRankTable)
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binaryStream;
    localRankTable.GetBinStream(true, binaryStream);
    binaryStream << currentStep_;

    // 字节流转换为vector<char>格式
    vector<char> sendMsg;
    binaryStream.Dump(sendMsg);

    // 发送
    socketAgent_.SendMsg(sendMsg.data(), sendMsg.size());

    HCCL_INFO("[RankInfoDetectClient::%s] end, currentStep_[%u].", __func__, currentStep_);
    currentStep_++;
}

void RankInfoDetectClient::ConstructSingleRank(RankTableInfo& localRankTable)
{
    localRankTable.version = "2.0";
    localRankTable.rankCount = 1;
    NewRankInfo rankInfo{};
    rankInfo.rankId = rankId_;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    CHK_PRT_CONT(GetLocalTlsStatus(rankInfo.tlsStatus), HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus"));
    localRankTable.ranks.emplace_back(rankInfo);

    // 打印
    localRankTable.Dump();
    HCCL_INFO(
        "[RankInfoDetectClient::%s] end, single rank, localRankTable[%s].", __func__,
        localRankTable.Describe().c_str());
}

void CheckRootInfoJson(const nlohmann::json& parseJson)
{
    // check version
    std::string version{};
    std::string msgVersion = "error occurs when parser rootinfo object of propName \"version\"";
    TRY_CATCH_THROW(InvalidParamsException, msgVersion, version = GetJsonProperty(parseJson, "version"););
    if (version != "2.0") {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({version, "version", "2.0"}));
        HCCL_ERROR("[%s] failed with version [%s] is not \"2.0\".", __func__, version.c_str());
        THROW<InvalidParamsException>("version error");
    }

    // parser topo_file_path
    std::string topoFilePath{};
    std::string msgRankTopoFile = "error occurs when parser object of propName \"topo_file_path\"";
    TRY_CATCH_THROW(InvalidParamsException, msgRankTopoFile,
                    topoFilePath = GetJsonProperty(parseJson, "topo_file_path"););

    // check topo_file_path
    char resolvedPath[PATH_MAX] = {0};
    bool isInvalidPath = (realpath(topoFilePath.c_str(), resolvedPath) == nullptr);
    if (isInvalidPath) {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({topoFilePath, "topo_file_path", "valid path"}));
        HCCL_ERROR("[%s] topo_file_path[%s] is not a valid real path", __func__, topoFilePath.c_str());
        THROW<InvalidParamsException>("topo_file_path error");
    }

    // parser rank_count
    u32 rankCount{};
    std::string msgRankcount = "error occurs when parser object of propName \"rank_count\"";
    TRY_CATCH_THROW(InvalidParamsException, msgRankcount, rankCount = GetJsonPropertyUInt(parseJson, "rank_count"););

    // parser rank_list
    nlohmann::json rankJsons{};
    std::string msgRanklist = "error occurs when parser object of propName \"rank_list\"";
    TRY_CATCH_THROW(InvalidParamsException, msgRanklist, GetJsonPropertyList(parseJson, "rank_list", rankJsons););

    // check rank_count
    bool isRankCountMismatch = (rankCount != rankJsons.size());
    if (isRankCountMismatch) {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(rankCount), "rankCount", std::to_string(rankJsons.size())}));
        HCCL_ERROR(
            "[%s] failed with rankCount is not equal to rank_list size."
            "rankCount[%u], ranks.size[%u]",
            __func__, rankCount, rankJsons.size());
        THROW<InvalidParamsException>("rankCount error");
    }
}

void RankInfoDetectClient::ConstructRankTable(RankTableInfo& localRankTable)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 单P场景处理
    CHK_PRT_RET((rankSize_ == 1), ConstructSingleRank(localRankTable), );

    // 1. 解析文件topoInfo.json
    std::string filePath = "/etc/hccl_rootinfo.json";
    JsonParser jsonParser{};
    nlohmann::json parseJson{};
    std::ifstream file(filePath);
    if (file.good()) {
        jsonParser.ParseFileToJson(filePath, parseJson);
    } else {
        size_t bufSize;
        s32 result = TopoAddrInfoGetSize(devPhyId_, &bufSize); // 获取rankInfo大小，用于提前分配内存
        CHK_PRT_THROW(
            result != 0 || bufSize > MAX_BUFFER_LEN,
            HCCL_ERROR("[RankInfoDetectClient::%s] Get rankinfo size failed.", __func__), InvalidParamsException,
            "Get rankinfo size failed.");
        std::vector<char> buffer(bufSize, '\0');
        result = TopoAddrInfoGet(devPhyId_, buffer.data(), &bufSize); // 获取rankInfo 并更新大小
        CHK_PRT_THROW(
            result != 0, HCCL_ERROR("[RankInfoDetectClient::%s] Get rankinfo failed.", __func__),
            InvalidParamsException, "Get rankinfo size failed.");
        std::string jsonString(buffer.data(), bufSize);
        // 将生成的info信息转换成json文件
        parseJson = nlohmann::json::parse(jsonString);
    }
    CheckRootInfoJson(parseJson);

    // 2. 获取当前devPhyId_对应的devInfo
    nlohmann::json localDevInfoJson{};
    GetLocalDevInfoJson(parseJson, localDevInfoJson);
    // 3. 在反序列化和上报本地 RankTable 前改写 addr，确保后续全局 RankTable 和 RankGraph 使用选中地址
    SelectLocalHostBackupAddr(localDevInfoJson);

    // 4. 组rankTable的json格式
    nlohmann::json localRankTableJson{};
    GetLocalRankTableJson(parseJson, localRankTableJson);
    localRankTableJson["rank_list"].push_back(localDevInfoJson); // 添加localDevInfoJson

    // 5. 反序列化获得RankTableInfo
    std::string msgDeserialize = "error occurs when localRankTable Deserialize";
    TRY_CATCH_THROW(InvalidParamsException, msgDeserialize, localRankTable.Deserialize(localRankTableJson, false););

    CHK_PRT_THROW(
        localRankTable.ranks.empty(), HCCL_ERROR("[RankInfoDetectClient::%s] local rank table has no rank.", __func__),
        InvalidParamsException, "local rank table has no rank");
    CHK_PRT_CONT(
        GetLocalTlsStatus(localRankTable.ranks[0].tlsStatus),
        HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus"));
    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

void RankInfoDetectClient::ProbeHostRoceAddr(const IpAddress& hostAddr, bool& isAvailable) const
{
    isAvailable = false;
    // hostAddr 来自 netLayer3 的 rank_addr_list，是 RoCE 数据通信地址；它不同于 RootInfoDetect
    // 使用的 rootHandle.ip，后者只负责控制面 socket 建链。
    EndpointDesc endpointDesc{};
    const HcommResult initRet = EndpointDescInit(&endpointDesc, 1);
    CHK_PRT_THROW(
        initRet != HCCL_SUCCESS, HCCL_ERROR("[%s] EndpointDescInit failed, ret[%d].", __func__, initRet),
        InternalException, StringFormat("[%s] EndpointDescInit failed, ret[%d]", __func__, initRet));
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    const std::string msgFillCommAddr = "fill host RoCE endpoint address failed";
    TRY_CATCH_THROW(InvalidParamsException, msgFillCommAddr, FillCommAddr(endpointDesc.commAddr, hostAddr););

    EndpointHandle endpointHandle = nullptr;
    const HcommResult createRet = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    // 仅网络错误允许上层继续尝试备用地址，其他错误按不可恢复异常立即终止。
    if (createRet == HCCL_E_NETWORK) {
        HCCL_WARNING(
            "[%s] host addr is unavailable, hostAddr[%s], ret[%d].", __func__, hostAddr.Describe().c_str(), createRet);
        return;
    } else if (createRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] HcommEndpointCreate failed, hostAddr[%s], ret[%d].", __func__, hostAddr.Describe().c_str(),
            createRet);
        THROW<InternalException>(StringFormat("[%s] HcommEndpointCreate failed, ret[%d]", __func__, createRet));
    }
    if (endpointHandle != nullptr) {
        // Endpoint 仅用于可用性探测，不参与后续通信，探测成功后立即释放。
        const HcommResult destroyRet = HcommEndpointDestroy(endpointHandle);
        CHK_PRT_THROW(
            destroyRet != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] HcommEndpointDestroy failed, hostAddr[%s], ret[%d].", __func__, hostAddr.Describe().c_str(),
                destroyRet),
            InternalException, StringFormat("[%s] HcommEndpointDestroy failed, ret[%d]", __func__, destroyRet));
    }
    isAvailable = true;
    HCCL_INFO(
        "[%s] host addr probe success, devPhyId[%u], rankId[%u], hostAddr[%s].", __func__, devPhyId_, rankId_,
        hostAddr.Describe().c_str());
}

void RankInfoDetectClient::SelectLocalHostBackupAddr(nlohmann::json& localDevInfoJson)
{
    const bool isLevelListInvalid = localDevInfoJson.empty() || !localDevInfoJson.contains("level_list")
                                    || !localDevInfoJson["level_list"].is_array();
    CHK_PRT_THROW(
        isLevelListInvalid,
        HCCL_ERROR(
            "[%s] level_list is missing or is not an array, devPhyId[%u], rankId[%u].", __func__, devPhyId_, rankId_),
        InvalidParamsException, "level_list is missing or is not an array");
    std::vector<nlohmann::json*> addrJsons;
    const std::string msgCollectLayer3Addr = "collect netLayer3 addr failed";
    TRY_CATCH_THROW(InvalidParamsException, msgCollectLayer3Addr, CollectLayer3AddrJsons(localDevInfoJson, addrJsons););
    if (addrJsons.empty()) {
        HCCL_DEBUG("[%s] no netLayer3+ addr with backup_addr needs probing.", __func__);
        return;
    }
    // 主备选择只依赖 RootInfo 的 net_layer 字段，不读取或构建 PhyTopo。
    // 对每个 netLayer3+ 地址独立探测主 addr；主地址不可用时，再按配置顺序逐个尝试 backup_addr。
    for (auto* addrJson : addrJsons) {
        SelectAvailableHostAddr(*addrJson);
    }
    HCCL_INFO(
        "[%s] end, devPhyId[%u], rankId[%u], addrConfigNum[%zu].", __func__, devPhyId_, rankId_, addrJsons.size());
}

void RankInfoDetectClient::SelectAvailableHostAddr(nlohmann::json& addrJson)
{
    std::vector<IpAddress> candidates;
    const std::string msgBuildCandidates = "build host addr candidates failed";
    TRY_CATCH_THROW(InvalidParamsException, msgBuildCandidates, BuildHostAddrCandidates(addrJson, candidates););
    HCCL_INFO(
        "[%s] devPhyId[%u], rankId[%u], primaryAddr[%s], backupAddrSize[%zu], "
        "candidateSize[%zu].",
        __func__, devPhyId_, rankId_, candidates.front().Describe().c_str(), candidates.size() - 1, candidates.size());

    // HCCL_E_NETWORK 是可恢复错误，通过 isAvailable 继续尝试下一个候选地址；
    // 其他异常不在此处恢复。
    for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
        bool isAvailable = false;
        ProbeHostRoceAddr(candidates[idx], isAvailable);
        if (isAvailable) {
            UpdateSelectedHostAddr(addrJson, candidates, idx);
            return;
        }
        if (idx == candidates.size() - 1) {
            THROW<NetworkApiException>(StringFormat("[%s] all host addr candidates are unavailable", __func__));
        }
        HCCL_WARNING(
            "[%s] host addr is unavailable, try next candidate, "
            "devPhyId[%u], rankId[%u], candidateAddr[%s], candidateIndex[%zu].",
            __func__, devPhyId_, rankId_, candidates[idx].Describe().c_str(), idx);
    }
}

void RankInfoDetectClient::UpdateSelectedHostAddr(
    nlohmann::json& addrJson, const std::vector<IpAddress>& candidates, std::size_t selectedIndex) const
{
    const std::string oldAddr = addrJson[ADDR_FIELD].get<std::string>();
    // 只改写当前有效 addr，保留 backup_addr 原始配置，并随本地 RankTable 一并上报。
    addrJson[ADDR_FIELD] = candidates[selectedIndex].GetIpStr();
    HCCL_RUN_INFO(
        "[%s] select host addr success, devPhyId[%u], rankId[%u], "
        "selectedNicRole[%s], oldHostAddr[%s], selectedHostAddr[%s], candidateIndex[%zu], tryCount[%zu].",
        __func__, devPhyId_, rankId_, selectedIndex == 0 ? "primary" : "backup", oldAddr.c_str(),
        candidates[selectedIndex].Describe().c_str(), selectedIndex, selectedIndex + 1);
}

void RankInfoDetectClient::GetLocalDevInfoJson(const nlohmann::json& parseJson, nlohmann::json& localDevInfoJson)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // rankList字段对应json内容
    nlohmann::json rankJsons;
    std::string msgRanklist = "error occurs when parser object of propName \"rank_list\"";
    TRY_CATCH_THROW(InvalidParamsException, msgRanklist, GetJsonPropertyList(parseJson, "rank_list", rankJsons););

    // 获取localrankJsons, 匹配deviceId字段与当前devPhyId_匹配的内容
    for (auto& rankJson : rankJsons) {
        u32 devId = 0;
        std::string msgDeviceId = "error occurs when parser object of propName \"device_id\"";
        TRY_CATCH_THROW(InvalidParamsException, msgDeviceId, devId = GetJsonPropertyUInt(rankJson, "device_id"););
        if (devId == devPhyId_) {
            HCCL_INFO("[RankInfoDetectClient::%s] find localDevInfoJson.", __func__);
            localDevInfoJson = rankJson;
            break;
        }
    }

    if (localDevInfoJson.empty()) {
        HCCL_ERROR("[%s] failed, no device_id matches devPhyId_[%u] in rank_list.", __func__, devPhyId_);
    }

    // 添加rankId
    localDevInfoJson["rank_id"] = rankId_;

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

void RankInfoDetectClient::GetLocalRankTableJson(const nlohmann::json& parseJson, nlohmann::json& localRankTableJson)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    std::string version;
    std::string msgVersion = "error occurs when parser object of propName \"version\"";
    TRY_CATCH_THROW(InvalidParamsException, msgVersion, version = GetJsonProperty(parseJson, "version"););
    localRankTableJson["version"] = version;

    std::string detour;
    std::string msgDetour = "error occurs when parser object of propName \"detour\"";
    TRY_CATCH_THROW(InvalidParamsException, msgDetour, detour = GetJsonProperty(parseJson, "detour", false););
    if (detour == "true") {
        localRankTableJson["detour"] = detour;
    }

    localRankTableJson["rank_count"] = rankSize_;
    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

void RankInfoDetectClient::RecvRankTableMsg(vector<char>& rankInfoMsg)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 接收数据
    u64 revMsgLen = 0;
    std::unique_ptr<HostBuffer> msg = std::make_unique<HostBuffer>(MAX_BUFFER_LEN);
    char* msgAddr = reinterpret_cast<char*>(msg->GetAddr());
    CHK_PRT_THROW(
        !socketAgent_.RecvMsg(msgAddr, revMsgLen),
        HCCL_ERROR("RankInfoDetectClient::%s, recv rankTable error.", __func__), SocketException, "client recv fail");

    // 以vector<char>格式保存
    rankInfoMsg.resize(revMsgLen);
    rankInfoMsg.assign(msgAddr, msgAddr + revMsgLen);

    HCCL_INFO("[RankInfoDetectClient::%s] end, revMsgLen[%llu].", __func__, revMsgLen);
}

// 解析接收到的rank table信息
void RankInfoDetectClient::ParseRankTable(vector<char>& rankInfoMsg)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 消息格式: [ranktable大小(u32, 4字节)][ranktable数据(n字节)][step(4字节)][failedAgentIdList]
    BinaryStream binStream(rankInfoMsg);

    // 解析localRankInfo
    rankTable_ = RankTableInfo(binStream);
    rankTable_.Dump();

    // 解析step
    u32 receivedStep;
    binStream >> receivedStep;

    // 解析failedAgentIdList
    std::string failedAgentIdList;
    binStream >> failedAgentIdList;
    if (failedAgentIdList.size() > 0) {
        // 建链失败时，打印 root 节点发来的临终遗言
        HCCL_ERROR(
            "[RankInfoDetectClient::%s] TopoDetect ERROR occur, failedRankIdList[%s]", __func__,
            failedAgentIdList.c_str());
    }

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

void RankInfoDetectClient::RecvRankTable()
{
    // 获取rankTable
    vector<char> rankInfoMsg{};
    RecvRankTableMsg(rankInfoMsg);

    // 解析rankTable
    ParseRankTable(rankInfoMsg);

    // 校验
    VerifyRankTable();
}

void RankInfoDetectClient::VerifyRankTable()
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 校验rankCount符合预期
    if (rankTable_.rankCount != rankSize_) {
        THROW<InvalidParamsException>(StringFormat(
            "[RankInfoDetectClient::%s] rank_count[%u] does not match"
            " rankSize_[%u].",
            __func__, rankTable_.rankCount, rankSize_));
    }

    // 校验rankTable内容
    rankTable_.Check();
    // TLS开关一致性校验
    HcclResult ret = VerifyTlsConsistency();
    CHK_PRT_THROW(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetectClient::%s] tls consistency verify failed, ret[%d]", __func__, ret),
        InvalidParamsException, "tls consistency verify failed");

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

HcclResult RankInfoDetectClient::GetLocalTlsStatus(TlsStatus& tlsStatus) const
{
    struct RaInfo raInfo;
    raInfo.mode = NetworkMode::NETWORK_OFFLINE;
    raInfo.phyId = devPhyId_;
    return HrtRaGetTlsStatus(&raInfo, tlsStatus);
}

void RankInfoDetectClient::GenerateTlsStatusStr(std::string& tlsStatusStr, const std::vector<u32>& tlsStatusRanks) const
{
    tlsStatusStr.clear();
    for (const auto& rank : tlsStatusRanks) {
        tlsStatusStr += std::to_string(rank) + ",";
    }
    if (!tlsStatusStr.empty() && tlsStatusStr.back() == ',') {
        tlsStatusStr.pop_back();
    }
}

void RankInfoDetectClient::ReportTlsConfigurationError(
    const std::string& tlsInconsistentTlsType, const std::string& tlsEnableRankStr,
    const std::string& tlsDisableRankStr, const std::string& tlsUnknownRankStr) const
{
    std::string expectMessage = "\"All ranks are consistent. Current status: rankList for enabled tls: "
                                + tlsEnableRankStr + "; rankList for disabled tls: " + tlsDisableRankStr
                                + "; rankList for query failure tls: " + tlsUnknownRankStr;
    std::string errormessage
        = "Value \"" + tlsInconsistentTlsType + "\" for config \"tls\" is invalid. Expected: " + expectMessage;

    RPT_INPUT_ERR(
        true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>({tlsInconsistentTlsType, "\"tls\"", expectMessage}));

    HCCL_ERROR("[ReportTlsConfigurationError][RanktableCheck] %s", errormessage.c_str());
}

HcclResult RankInfoDetectClient::VerifyTlsConsistency() const
{
    bool isSupportCheckTlsStatus = true; // 用于标识是否存在不支持查询Tls开关状态的情况
    bool isTlsConsistent = true;         // 用于标识TLS开关状态是否一致
    std::vector<u32> tlsEnableRank;
    std::vector<u32> tlsDisableRank;
    std::vector<u32> tlsUnknownRank;

    for (const auto& rankInfo : rankTable_.ranks) {
        if (rankInfo.tlsStatus == TlsStatus::ENABLE) {
            tlsEnableRank.push_back(rankInfo.rankId);
        } else if (rankInfo.tlsStatus == TlsStatus::DISABLE) {
            tlsDisableRank.push_back(rankInfo.rankId);
        } else {
            isSupportCheckTlsStatus = false;
            tlsUnknownRank.push_back(rankInfo.rankId);
        }
    }

    // 将卡的信息汇总成string
    std::string tlsEnableRankStr;
    std::string tlsDisableRankStr;
    std::string tlsUnknownRankStr;
    GenerateTlsStatusStr(tlsEnableRankStr, tlsEnableRank);
    GenerateTlsStatusStr(tlsDisableRankStr, tlsDisableRank);
    if (!isSupportCheckTlsStatus) {
        GenerateTlsStatusStr(tlsUnknownRankStr, tlsUnknownRank);
    }

    std::string tlsInconsistentTlsType;
    if (!tlsEnableRank.empty() && !tlsDisableRank.empty()) {
        isTlsConsistent = false;
        tlsInconsistentTlsType = (tlsDisableRank.size() <= tlsEnableRank.size()) ? "Disable" : "Enable";
    }

    // 四种不同情况
    if (isTlsConsistent && isSupportCheckTlsStatus) {
        // 1.通信域所有卡都支持查询TLS开关状态，并且TLS开关状态都是一致的。
        HCCL_INFO("[Verify][TlsConsistency] All ranks tlsStatus are consistent");
    } else if (!isTlsConsistent && isSupportCheckTlsStatus) {
        // 2.通信域所有卡都支持查询TLS开关状态，但是TLS开关状态存在不一致，报错。
        ReportTlsConfigurationError(tlsInconsistentTlsType, tlsEnableRankStr, tlsDisableRankStr, tlsUnknownRankStr);
        return HCCL_E_PARA;
    } else if (isTlsConsistent && !isSupportCheckTlsStatus) {
        // 3.通信域内的部分卡不支持查询TLS开关状态，目前能查询到的卡的TLS开关状态是一致的，打印warning提醒
        HCCL_WARNING(
            "[Verify][TlsConsistency] Some ranks do not support to check tlsStatus, "
            "not support rankId: [%s]",
            tlsUnknownRankStr.c_str());
    } else {
        // 4.通信域内的部分卡不支持查询TLS开关状态，但是目前能查询到的卡的TLS开关状态已经不一致，报错
        ReportTlsConfigurationError(tlsInconsistentTlsType, tlsEnableRankStr, tlsDisableRankStr, tlsUnknownRankStr);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

void RankInfoDetectClient::HostListenPortDetect(NewRankInfo& rankInfo)
{
    std::string topoPath = GetRootInfoTopoFilePath();
    PhyTopoBuilder::GetInstance().Build(topoPath);
    auto devLogicId = HrtGetDevice();
    u32 devPhyId = rankInfo.deviceId;
    for (auto& rankLevelInfo : rankInfo.rankLevelInfos) {
        // 物理图不区分逻辑层，按 RankTable 端口匹配当前层。
        auto graph = PhyTopo::GetInstance()->GetTopoGraph();
        if (graph == nullptr) {
            HCCL_DEBUG("[RankInfoDetectClient::%s] Physical topo graph is nullptr.", __func__);
            continue;
        }
        std::vector<std::shared_ptr<PhyTopo::Link>> links = graph->GetEdges(rankInfo.localId);
        for (auto& link : links) {
            if (!IsHostRdmaLink(link)) {
                continue;
            }
            const auto& phyPorts = link->GetSourceIFace()->GetPorts();
            // 使用物理 Host RDMA 端口对应的地址，不默认取首个地址。
            auto addrIt = std::find_if(
                rankLevelInfo.rankAddrs.begin(), rankLevelInfo.rankAddrs.end(),
                [&phyPorts](const AddressInfo& addrInfo) {
                    return HasMatchingPort(addrInfo, phyPorts);
                });
            if (addrIt == rankLevelInfo.rankAddrs.end()) {
                continue;
            }
            HCCL_DEBUG("[SocketManager::%s] find the host rdma link %s", __func__, link->Describe().c_str());
            const IpAddress& hostIp = addrIt->addr;
            uint32_t hostPort = 0;
            SetupHostListenPort(devLogicId, devPhyId, hostIp, hostPort);
            rankInfo.hostPort = hostPort;
            return;
        }
    }
}

void RankInfoDetectClient::SetupHostListenPort(
    u32 devLogicId, u32 devPhyId, const IpAddress& hostIp, uint32_t& hostPort)
{
    std::lock_guard<std::mutex> lock(hostSocketLock_);
    u32 listenPort = HCCL_INVALID_PORT;
    auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
    u32 basePort = EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort();
    if (portRange.empty() && basePort != HCCL_INVALID_PORT) {
        listenPort = basePort + devPhyId;
        HCCL_INFO("[RankInfoDetectClient::%s] BasePort is configured, listenPort[%u].", __func__, listenPort);
        hostPort = listenPort;
        return;
    }

    if (portRange.empty()) {
        constexpr u32 HOST_CONTROL_BASE_PORT = 60000; // 控制面起始port
        HCCL_INFO(
            "[RankInfoDetectClient::%s] No port configuration, using default port range[%u, %u]", __func__,
            HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT);
        SocketPortRange defaultRange = {HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT};
        portRange.push_back(defaultRange);
    }

    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId, hostIp);
    hostSocket_ = std::make_shared<Socket>(
        hostSocketHandle, hostIp, HCCL_INVALID_PORT, hostIp, "hostport_preempt", SocketRole::SERVER,
        NicType::HOST_NIC_TYPE);
    PreemptPortManager::GetInstance(devLogicId).ListenPreempt(hostSocket_, portRange, listenPort);
    HCCL_INFO("[RankInfoDetectClient::%s] preempt hostPort[%u] success.", __func__, listenPort);

    // 登记到进程级 map，供算子下发阶段复用，避免跨阶段端口竞争
    DevNetPortType portType(ConnectProtoType::RDMA);
    PortData portData(static_cast<RankId>(devPhyId), portType, 0, hostIp);
    SocketManager socketMgr(0, devPhyId, static_cast<u32>(devLogicId), "hostport_preempt");
    hostSocketRegistered_ = socketMgr.RegisterHostListenSocket(portData, hostSocket_);

    hostPort = listenPort;
}

void RankInfoDetectClient::SocketTearDown(u32 devPhyId)
{
    std::lock_guard<std::mutex> lock(hostSocketLock_);
    if (hostSocket_ == nullptr) {
        return;
    }
    const IpAddress& hostIp = hostSocket_->GetLocalIp();
    auto devLogicId = HrtGetDevice();
    if (hostSocketRegistered_) {
        // 已登记到 SocketManager::GetServerSocketMap()，所有权已转移给 map，
        // 由算子下发阶段 HostSocketStopListen refcount 归 0 时清理，此处跳过 Release/Destroy
        HCCL_INFO(
            "[RankInfoDetectClient::%s] hostSocket already registered to ServerSocketMap, "
            "skip Release/Destroy, only release local ref.",
            __func__);
    } else if (
        EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange().size() > 0
        || EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort() == HCCL_INVALID_PORT) {
        // 若开启抢占监听端口
        PreemptPortManager::GetInstance(devLogicId).Release(hostSocket_);
        HostSocketHandleManager::GetInstance().Destroy(devPhyId, hostIp);
    }
    hostSocket_ = nullptr;
}

void RankInfoDetectClient::TearDown()
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);
    SocketTearDown(devPhyId_);

    // close socket
    clientSocket_->Close();

    // deinit handle
    HostSocketHandleManager::GetInstance().Destroy(devPhyId_, clientSocket_->GetLocalIp());

    // deinit ra in detach thread to avoid block main thread
    s32 deviceLogicId = HrtGetDevice();
    std::thread{[deviceLogicId]() {
        EXCEPTION_CATCH(
            HccpPeerManager::GetInstance().DeInit(deviceLogicId),
            HCCL_ERROR("[RankInfoDetectClient::TearDown] DeInit exception"));
    }}.detach();

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

RankInfoDetectClient::~RankInfoDetectClient() { DECTOR_TRY_CATCH("RankInfoDetectClient", TearDown()); }

} // namespace Hccl
