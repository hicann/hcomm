/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableStandard.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

// ltm指定config路径
#include "common/src/config.h"
#include "hccl_comm_pub.h"
#include "workflow_pub.h"

using namespace std;
using namespace hccl;

TopoinfoRanktableStandard::TopoinfoRanktableStandard(const std::string& rankTableM, const std::string& identify)
    : TopoInfoRanktableParser(rankTableM, identify)
{}

TopoinfoRanktableStandard::~TopoinfoRanktableStandard() {}

HcclResult TopoinfoRanktableStandard::Init() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableStandard::GetSelfClusterInfo(HcclCommParams& params) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableStandard::GetClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}
HcclResult TopoinfoRanktableStandard::GetClusterInfo(RankTable_t& clusterInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableStandard::ParserClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetHcomInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetServerList(
    const nlohmann::json& obj, u32 objIndex, hccl::RankTable_t& rankTable, u32 serverNum)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetSingleServer(
    const nlohmann::json& serverListObj, u32 objIndex, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetCloudHcomInfo(
    hccl::HcclCommParams& params, hccl::RankTable_t& rankTable, const std::string& identify, u32& rank)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetSortClouldRankList(hccl::RankTable_t& rankTable) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableStandard::GetSingleGroupDeviceCount(
    nlohmann::json& obj, u32 objIndex, hccl::RankTable_t& rankTable, u32& deviceNum)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetLabSingleGroup(
    nlohmann::json& obj, u32 objIndex, hccl::HcclCommParams& params, hccl::RankTable_t& rankTable, u32 instanceNum)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetGroupList(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetInstanceList(
    nlohmann::json& instanceList, hccl::HcclCommParams& params, hccl::RankTable_t& rankTable, u32 instanceNum,
    u32 deviceNum)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetCloudDevList(
    nlohmann::json& instanceList, u32 podIndex, nlohmann::json& deviceList, std::string& serverId, u32& serverIdx)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetDevList(
    nlohmann::json& instanceList, u32 podIndex, nlohmann::json& deviceList, hccl::HcclCommParams& params,
    hccl::RankTable_t& rankTable, std::string& serverId, u32& serverIdx)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetDeployMode(bool& cloudFlag) const { return HCCL_E_NOT_SUPPORT; }
