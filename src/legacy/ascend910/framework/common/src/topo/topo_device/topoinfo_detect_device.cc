/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_detect.h"
#include <string>
#include "adapter_rts_common.h"
#include "hccl_whitelist.h"
#include "hccl_socket.h"
#include "sal_pub.h"
#include "device_capacity.h"
#include "preempt_port_manager.h"

using namespace std;
namespace hccl {
UniversalConcurrentMap<u32, volatile u32> TopoInfoDetect::g_topoExchangeServerStatus_;

TopoInfoDetect::TopoInfoDetect()
    : deviceLogicID_(INVALID_INT),
      localRankInfo_(),
      clusterTopoInfo_(),
      isInterSuperPodRetryEnable_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoInfoDetect::~TopoInfoDetect() {}

HcclResult TopoInfoDetect::GetServerConnections(std::map<u32, std::shared_ptr<HcclSocket>>& connectSockets)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetAgentListenSocket(HcclSocketPortConfig& commPortConfig) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::GetAgentConnection(std::shared_ptr<HcclSocket>& connectSocket) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::SendGroupLeaderPort(std::shared_ptr<HcclSocket>& connectSocket, HcclRankHandle& rankHandle)
{
    return HCCL_E_NOT_SUPPORT;
}

void TopoInfoDetect::SetupTopoGroupLeader(
    s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort, vector<HcclIpAddress> whitelist,
    HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket, std::shared_ptr<HcclSocket> grpLeaderToRoot,
    bool isMasterInfo)
{
    return;
}

void TopoInfoDetect::SetupTopoExchangeServer(
    s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort, vector<HcclIpAddress> whitelist,
    HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket, bool isMasterInfo)
{
    return;
}
HcclResult
TopoInfoDetect::SetupServerByMasterInfo(const HcclIpAddress& masterIP, u32 masterPort, const HcclRootHandle& rootInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupServer(HcclRootHandle& rootInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::GroupLeaderListen(HcclRankHandle& rankHandle, vector<HcclIpAddress>& whitelist)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GroupLeaderAccept(
    HcclRankHandle& grpLeaderInfo, vector<HcclIpAddress> whitelist, std::shared_ptr<HcclSocket> grpLeaderToRoot)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GenerateRootInfo(
    const HcclIpAddress& hostIP, u32 hostPort, u32 devicePhysicID, HcclRootHandle& rootInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::CalcGroupSizeAndRank(const u32 nRanks, const u32 rank, u32& groupSize, u32& groupRank)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupGroupMember(u32 rankSize, u32 myrank, const HcclRootHandle& rootInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::TeardownServer() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::WaitTopoExchangeServerCompelte(u32 idx) const { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::PrepareHandle(HcclRankHandle& rankHandle, std::vector<HcclIpAddress>& whitelist)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupAgent(
    u32 rankSize, u32 myrank, const HcclRootHandle& rootInfo, const HcclRankHandle& rankHandle,
    const CommConfig& commConfig)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupRank(std::shared_ptr<HcclSocket>& agentConnRoot) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::TeardownAgent() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::SetupAgentByMasterInfo(HcclIpAddress& localHostIp, const HcclRootHandle& rootInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::WaitComplete(const HcclRootHandle& rootInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::Teardown() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::ReadHostSocketWhitelist(vector<HcclIpAddress>& whitelist) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetAllHostIfInfos(vector<pair<string, HcclIpAddress>>& ifInfos, u32 devPhyId) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetAllValidHostIfInfos(
    const vector<HcclIpAddress>& whitelist, vector<pair<string, HcclIpAddress>>& ifInfos, u32 devPhyId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetRootHostIP(const vector<HcclIpAddress>& whitelist, HcclIpAddress& ip, u32 devPhyId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetGroupLeader(HcclRankHandle& rankHandle) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::SetIsInterSuperPodRetryEnable(bool isRetry) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::StartRootNetwork(
    const HcclIpAddress& hostIP, u32& usePort, const std::vector<HcclSocketPortRange>& portRanges)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::StartGroupLeaderNetwork(
    const vector<HcclIpAddress>& whitelist, const HcclIpAddress& hostIP, u32& bindPort,
    const std::vector<HcclSocketPortRange>& portRanges)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::AddSocketWhiteList(u32 port, const vector<HcclIpAddress>& whitelist) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::StartNetwork(HcclIpAddress& hostIP, bool bInitDevNic) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::StopNetwork(HcclIpAddress& hostIP, bool bInitDevNic) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::FilterDevIPs(
    std::vector<HcclIpAddress>& sourceDeviceIPs, std::vector<HcclIpAddress>& targetDeviceIPs) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::PreemptDeviceNicPort(
    const u32 devPhyId, const s32 devLogicId, const HcclIpAddress& deviceIp, u32& usePort)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::PreemptDeviceVnicPort(HcclBasicRankInfo& localRankInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::PreemptBackupDeviceNicPort(
    const u32 devPhyId, const s32 devLogicId, const HcclIpAddress& deviceIp, const HcclIpAddress& backupDeviceIp,
    u32& usePort)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetDeviceBackupNicInfo(HcclBasicRankInfo& localRankInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::GenerateLocalRankInfo(u32 rankSize, u32 rankID, HcclBasicRankInfo& localRankInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetSuperPodInfo(s32 deviceLogicId, std::string& superPodId, u32& superDeviceId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetCluterInfo(RankTable_t& clusterInfo) { return HCCL_E_NOT_SUPPORT; }
HcclResult TopoInfoDetect::GetRankId(u32& rankId)
{
    rankId = identifierNum_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetLocalRankInfo(HcclBasicRankInfo& rankInfo) { return HCCL_E_NOT_SUPPORT; }

void TopoInfoDetect::SetBootstrapHostIP(HcclIpAddress& ip) { bootstrapHostIP_ = ip; }

HcclIpAddress TopoInfoDetect::GetBootstrapHostIP() const { return bootstrapHostIP_; }
HcclResult TopoInfoDetect::TransformRankTableStr(const RankTable_t& clusterInfo, string& ranktableStr)
{
    return HCCL_E_NOT_SUPPORT;
}
HcclResult TopoInfoDetect::TransformDeviceList(
    const RankTable_t& clusterInfo, vector<RankInfo_t>& tmpRankList, nlohmann::json& perServerJson, u32 serverIndex)
{
    return HCCL_E_NOT_SUPPORT;
}
HcclResult TopoInfoDetect::Struct2JsonRankTable(const RankTable_t& clusterInfo, nlohmann::json& ClusterJson)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TopoInfoDetect::TransformSuperPodList(const std::vector<RankInfo_t>& rankInfo, nlohmann::json& superPodListJson) const
{
    return HCCL_E_NOT_SUPPORT;
}
} // namespace hccl
