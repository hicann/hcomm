/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_roletableParser.h"

#include "externalinput_pub.h"
#include "hccl_comm_pub.h"

using namespace std;
using namespace hccl;

TopoinfoRoletable::TopoinfoRoletable(const std::string& rankTableM) : TopoInfoRanktableParser(rankTableM, "0") {}

TopoinfoRoletable::~TopoinfoRoletable() {}

HcclResult
TopoinfoRoletable::GetSingleNode(const nlohmann::json& NodeListObj, u32 objIndex, std::vector<RoleTableNodeInfo>& nodes)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRoletable::GetServersInfo(std::vector<RoleTableNodeInfo>& servers) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRoletable::GetClientsInfo(std::vector<RoleTableNodeInfo>& clients) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRoletable::ParserRoleTable(RoleTableInfo& roleTableInfo) { return HCCL_E_NOT_SUPPORT; }
