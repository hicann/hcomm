/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_team_c_adpt.h"
#include "hcomm_team.h"
#include "hcomm_team_mgr.h"
#include "hcomm_result_defs.h"
#include "hcomm_team_entity_defs.h"
#include "log.h"

using namespace hcomm;

HcommResult HcommTeamCreate(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* syncMemSize)
{
    CHK_PRT_RET(
        (desc == nullptr || team == nullptr || syncMemSize == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__),
        HCOMM_E_PTR);
    CHK_PRT_RET(
        (desc->memberNum == 0 || desc->memberNum == 1), HCCL_ERROR("[%s] memberNum cannot be 0 or 1", __func__),
        HCOMM_E_PARA);
    CHK_PRT_RET(
        desc->selfMemberId >= desc->memberNum,
        HCCL_ERROR("[%s] selfMemberId[%u] >= memberNum[%u]", __func__, desc->selfMemberId, desc->memberNum),
        HCOMM_E_PARA);
    CHK_PRT_RET(
        (desc->requirement.signalCount != 0 || desc->requirement.counterCount != 0),
        HCCL_ERROR(
            "[%s] signalCount[%u] and counterCount[%u] must be 0", __func__, desc->requirement.signalCount,
            desc->requirement.counterCount),
        HCOMM_E_PARA);
    CHK_PRT_RET(
        desc->requirement.barrierCount == 0, HCCL_ERROR("[%s] barrierCount must be >= 1", __func__), HCOMM_E_PARA);
    CHK_PRT_RET(
        worldTeam != nullptr && desc->worldMemberIds == nullptr,
        HCCL_ERROR("[%s] sub team worldMemberIds is null", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().TeamCreate(worldTeam, desc, team, syncMemSize);
}

HcommResult HcommTeamDestroy(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[%s] team is nullptr", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().TeamDestroy(team);
}

HcommResult HcommTeamBindChannels(HcommTeamHandle team, const HcommTeamBindChannelsDesc* desc)
{
    if (team == nullptr || desc == nullptr || desc->channelNumPerMember == nullptr
        || desc->channelsByMemberId == nullptr) {
        HCCL_ERROR("[%s] nullptr parameter", __func__);
        return HCOMM_E_PTR;
    }
    return HcommTeamMgr::GetInstance().BindChannels(team, desc);
}

HcommResult HcommTeamBindRemoteSyncMem(HcommTeamHandle team, const HcommTeamBindSyncMemDesc* remoteDesc)
{
    CHK_PRT_RET(
        (team == nullptr || remoteDesc == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(remoteDesc->remoteMems == nullptr, HCCL_ERROR("[%s] remoteMems is nullptr", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(remoteDesc->remoteMemNum == 0, HCCL_ERROR("[%s] remoteMemNum is zero", __func__), HCOMM_E_PARA);
    return HcommTeamMgr::GetInstance().BindSyncMem(team, remoteDesc);
}

HcommResult HcommTeamWindowRegister(
    HcommTeamHandle worldTeam, const HcommTeamWindowDesc* desc, HcommWindowHandle* handle, HcommTeamWindowFlag flag)
{
    CHK_PRT_RET(
        (worldTeam == nullptr || handle == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(
        flag != HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC,
        HCCL_ERROR("[%s] flag[%d] is not supported, only support 0", __func__, flag), HCOMM_E_PARA);
    /* desc 当前由后续 HcommTeamWindowBindRemoteMems 单独绑定 window 的 mems，注册阶段不消费，允许为 nullptr。 */
    (void)desc;
    return HcommTeamMgr::GetInstance().WindowRegister(worldTeam, handle);
}

HcommResult
HcommTeamWindowBindRemoteMems(HcommTeamHandle team, HcommWindowHandle handle, const HcommTeamWindowDesc* remoteDesc)
{
    if (team == nullptr || handle == nullptr || remoteDesc == nullptr || remoteDesc->mems == nullptr) {
        HCCL_ERROR("[%s] nullptr parameter", __func__);
        return HCOMM_E_PTR;
    }
    return HcommTeamMgr::GetInstance().BindWindow(team, handle, remoteDesc);
}

HcommResult HcommTeamWindowDeregister(HcommTeamHandle worldTeam, HcommWindowHandle handle)
{
    CHK_PRT_RET(
        (worldTeam == nullptr || handle == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().WindowDeregister(worldTeam, handle);
}

HcommResult HcommTeamGetNetLayer(HcommTeamHandle team, uint32_t* netLayer)
{
    CHK_PRT_RET((team == nullptr || netLayer == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().GetNetLayer(team, netLayer);
}

HcommResult HcommTeamIsSubBelongToWorld(HcommTeamHandle worldTeam, HcommTeamHandle subTeam, bool* isBelong)
{
    CHK_PTR_NULL(worldTeam);
    CHK_PTR_NULL(subTeam);
    CHK_PTR_NULL(isBelong);
    HCCL_ERROR("[%s] not support", __func__);
    return HCOMM_E_NOT_SUPPORT;
}
