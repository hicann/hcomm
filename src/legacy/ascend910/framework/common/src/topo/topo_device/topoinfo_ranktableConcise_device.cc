/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableConcise.h"
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "log.h"
#include "env_config.h"
#include "hccl_comm_pub.h"
// ltm指定config路径
#include "common/src/config.h"
#include "workflow_pub.h"
#include "device_capacity.h"

using namespace std;
using namespace hccl;

TopoinfoRanktableConcise::TopoinfoRanktableConcise(const std::string& rankTableM, const std::string& identify)
    : TopoInfoRanktableParser(rankTableM, identify),
      isInterSuperPodRetryEnable_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoinfoRanktableConcise::~TopoinfoRanktableConcise() {}

HcclResult TopoinfoRanktableConcise::Init() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableConcise::GetClusterInfo(RankTable_t& clusterInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableConcise::GetSelfClusterInfo(HcclCommParams& params) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableConcise::GetClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::SetIsInterSuperPodRetryEnable(bool isRetryEnable) { return HCCL_E_NOT_SUPPORT; }

void TopoinfoRanktableConcise::DetectNicDepoly(RankTable_t& rankTable) { return; }

HcclResult TopoinfoRanktableConcise::ParserClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::CheckNicDeployConsistence(RankTable_t& clusterInfo, NICDeployment deploy) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetRanktableInfo(RankTable_t& clusterInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableConcise::GetServerList(const nlohmann::json& obj, RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TopoinfoRanktableConcise::GetSingleServer(const nlohmann::json& serverListObj, u32 objIndex, RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetDeviceList(
    const nlohmann::json& serverListObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& hostIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleDevice(
    const nlohmann::json& deviceListObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& hostIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::SplitString(
    const std::string& str, const std::string& strC, std::vector<std::string>& strVector) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleDeviceIp(
    const nlohmann::json& deviceListObj, u32 objIndex, RankTable_t& clusterInfo, RankInfo_t& rankinfo,
    DevType deviceType, bool invalidHostIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleBackupDeviceIp(
    const nlohmann::json& deviceListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleDeviceHostPort(
    const nlohmann::json& deviceListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TopoinfoRanktableConcise::GetSingleDevicePort(const nlohmann::json& deviceListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleBackupDevicePort(
    const nlohmann::json& deviceListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::VerifyBackupDeviceIpAndPort(std::vector<RankInfo_t>& rankList, u32 devIndex)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperDeviceId(
    const nlohmann::json& deviceListObj, u32 objIndex, RankTable_t& clusterInfo, RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSuperPodList(const nlohmann::json& obj, RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TopoinfoRanktableConcise::GetSingleSuperPod(const nlohmann::json& superPodList, u32 objIndex, RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSuperPodServerList(
    const nlohmann::json& superPodList, u32 objIndex, RankTable_t& clusterInfo, std::string superPodId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperPodSever(
    const nlohmann::json& superPodServerList, u32 objIndex, RankTable_t& clusterInfo, std::string superPodId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::CheckSuperPodInfo(RankTable_t& clusterInfo) const { return HCCL_E_NOT_SUPPORT; }

HcclResult
TopoinfoRanktableConcise::GetSingleNicInfo(const nlohmann::json& serverListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}
