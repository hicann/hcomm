/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableHeterog.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "externalinput_pub.h"
#include "hccl_comm_pub.h"
// ltm指定config路径
#include "common/src/config.h"
#include "workflow_pub.h"

using namespace std;
using namespace hccl;

TopoinfoRanktableHeterog::TopoinfoRanktableHeterog(
    const std::string& rankTableM, const std::string& identify, DevType deviceType)
    : TopoInfoRanktableParser(rankTableM, identify),
      deviceType_(deviceType)
{}

TopoinfoRanktableHeterog::~TopoinfoRanktableHeterog() {}

HcclResult TopoinfoRanktableHeterog::Init() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableHeterog::GetSelfClusterInfo(HcclCommParams& params) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableHeterog::GetClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetClusterInfo(RankTable_t& clusterInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableHeterog::ParserClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetRanktableInfo(RankTable_t& clusterInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableHeterog::CheckNicDeployConsistence(RankTable_t& clusterInfo) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::CheckMode(std::string& mode) const { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableHeterog::CheckHeterogSubVersion(std::string& subVersion) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetHostPort(const u32& localRank, u32& hostPort) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableHeterog::GetRanks(
    const nlohmann::json& NodeListObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& nodeIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TopoinfoRanktableHeterog::GetSingleNode(const nlohmann::json& NodeListObj, u32 objIndex, RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

// 91093暂定所有字段都是必选字段。除了ranks里面的 bind_device_id
HcclResult TopoinfoRanktableHeterog::GetSingleRank91093(
    const nlohmann::json& ranksObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& nodeIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetSingleRank(
    const nlohmann::json& ranksObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& nodeIp)
{
    return HCCL_E_NOT_SUPPORT;
}
