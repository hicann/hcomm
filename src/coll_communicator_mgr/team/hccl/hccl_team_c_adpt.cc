/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_team_c_adpt.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "adapter_rts_common.h"
#include "hccl_comm_pub.h"
#include "hccl_team.h"
#include "hccl_team_mgr.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm_team.h"
#include "hcomm_team_c_adpt.h"
#include "hcomm_team_mgr.h"

using namespace hccl;

/**
 * @note 职责：集合通信的通信域HcclTeam管理的C接口的C到C++适配
 */

/* 在 rankIds 中查找 selfRankId 的下标作为 memberId。
 * 例如 rankIds=[1,3,5,7]，selfRankId=3，则 selfMemberId=1。未找到返回 HCCL_E_NOT_FOUND。 */
static HcclResult FindSelfMemberId(const HcclTeamCreateDesc* desc, uint32_t& selfMemberId)
{
    for (uint32_t i = 0; i < desc->rankNum; i++) {
        if (desc->rankIds[i] == desc->selfRankId) {
            selfMemberId = i;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[FindSelfMemberId] selfRankId[%u] not found in rankIds", desc->selfRankId);
    return HCCL_E_NOT_FOUND;
}

/* 从 HcclTeamCreateDesc 填充 HcommTeamCreateDesc。
 * worldMemberIds：worldTeam 传 nullptr（L3 生成 [0,memberNum)）；subTeam 传 worldTeam 的 memberId 列表。 */
static void FillHcommTeamCreateDesc(
    const HcclTeamCreateDesc* desc, uint32_t selfMemberId, const uint32_t* worldMemberIds,
    HcommTeamCreateDesc& hcommDesc)
{
    /* 用头文件 Init 初始化 ABI 头部（magicWord/version/size）与默认字段，再覆盖业务字段。 */
    (void)HcommTeamCreateDescInit(&hcommDesc);
    hcommDesc.memberNum = desc->rankNum;
    hcommDesc.selfMemberId = selfMemberId;
    hcommDesc.worldMemberIds = worldMemberIds;
    hcommDesc.requirement.signalCount = 0;  // 不支持配置，强制为0
    hcommDesc.requirement.counterCount = 0; // 不支持配置，强制为0
    hcommDesc.requirement.barrierCount = desc->requirement.barrierCount;
    hcommDesc.netLayer = desc->netLayer;
    hcommDesc.protocol = desc->protocol;
}

/* 反查 subTeam.rankIds 在 worldTeam.rankIds 中的下标作为 worldMemberIds（worldTeam 的 memberId）。
 * 用 unordered_map 预建 rankId→worldMemberId 反查表，O(N+M) 避免双重循环。 */
static HcclResult BuildSubTeamWorldMemberIds(
    HcommTeamHandle worldTeam, const uint32_t* subRankIds, uint32_t subRankNum, std::vector<uint32_t>& worldMemberIds)
{
    std::vector<uint32_t> worldRankIds = HcclTeamMgr::GetInstance().GetRankIds(worldTeam);
    CHK_PRT_RET(
        worldRankIds.empty(), HCCL_ERROR("[%s] GetRankIds failed, worldTeam[%p] not registered", __func__, worldTeam),
        HCCL_E_PARA);
    std::unordered_map<uint32_t, uint32_t> rankToMember;
    rankToMember.reserve(worldRankIds.size());
    for (uint32_t m = 0; m < worldRankIds.size(); m++) {
        rankToMember[worldRankIds[m]] = m;
    }
    worldMemberIds.resize(subRankNum);
    for (uint32_t i = 0; i < subRankNum; i++) {
        auto it = rankToMember.find(subRankIds[i]);
        CHK_PRT_RET(
            it == rankToMember.end(),
            HCCL_ERROR("[%s] subTeam rankId[%u] not in worldTeam rankIds", __func__, subRankIds[i]), HCCL_E_PARA);
        worldMemberIds[i] = it->second;
    }
    return HCCL_SUCCESS;
}

/* 申请 syncMem 本地内存。失败时回滚已创建的 team（HcommTeamDestroy）。 */
static HcclResult AllocTeamSyncMem(HcommTeamHandle* team, uint64_t syncMemSize, void*& syncMemPtr)
{
    syncMemPtr = nullptr;
    HcclResult mallocRet = hrtMalloc(&syncMemPtr, syncMemSize);
    if (mallocRet != HCCL_SUCCESS || syncMemPtr == nullptr) {
        HCCL_ERROR("[AllocTeamSyncMem] hrtMalloc failed, ret[%d] size[%llu]", mallocRet, syncMemSize);
        (void)HcommTeamDestroy(*team);
        *team = nullptr;
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclWorldTeamCreate(HcclComm comm, const HcclTeamCreateDesc* desc, HcommTeamHandle* worldTeam)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(desc);
    CHK_PTR_NULL(worldTeam);
    CHK_PRT_RET(
        (desc->rankNum == 0 || desc->rankNum == 1),
        HCCL_ERROR("[%s] world team can not be empty or single rank", __func__), HCCL_E_PARA);
    CHK_PRT_RET(desc->rankIds == nullptr, HCCL_ERROR("[%s] rankIds is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        (desc->requirement.signalCount != 0 || desc->requirement.counterCount != 0),
        HCCL_ERROR(
            "[%s] signalCount[%u] and counterCount[%u] must be 0", __func__, desc->requirement.signalCount,
            desc->requirement.counterCount),
        HCCL_E_PARA);
    CHK_PRT_RET(
        desc->requirement.barrierCount == 0, HCCL_ERROR("[%s] barrierCount must be >= 1", __func__), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    const std::string commId = collComm->GetCommId();

    /* 0. 校验 desc 与通信域匹配（worldTeam 可为子集）。 */
    uint32_t commRankSize = collComm->GetRankSize();
    CHK_PRT_RET(
        desc->rankNum > commRankSize,
        HCCL_ERROR(
            "[%s] rankNum[%u] > commRankSize[%u], comm[%s]", __func__, desc->rankNum, commRankSize, commId.c_str()),
        HCCL_E_PARA);

    /* 1. selfRankId 是本 rank 的实际 rankId，需在 rankIds 中查找其下标作为 memberId。 */
    uint32_t selfMemberId = 0;
    CHK_RET(FindSelfMemberId(desc, selfMemberId));

    /* 2. 填充 HcommTeamCreateDesc，world team 无父 team，worldMemberIds 传 nullptr（L3 生成 [0,memberNum)）。 */
    HcommTeamCreateDesc hcommDesc = {};
    FillHcommTeamCreateDesc(desc, selfMemberId, nullptr, hcommDesc);

    /* 3. 调用 Hcomm 层创建 world team，outSyncMemSize 为需要本地申请的 syncMem 字节数。 */
    uint64_t syncMemSize = 0;
    HcommResult ret = HcommTeamCreate(nullptr, &hcommDesc, worldTeam, &syncMemSize);
    CHK_PRT_RET(
        ret != 0 || *worldTeam == nullptr,
        HCCL_ERROR(
            "[%s] HcommTeamCreate failed, comm[%s] ret[%d] rankNum[%u] selfRankId[%u]", __func__, commId.c_str(), ret,
            desc->rankNum, desc->selfRankId),
        (ret != 0) ? static_cast<HcclResult>(ret) : HCCL_E_INTERNAL);
    CHK_PRT_RET(syncMemSize == 0, HCCL_ERROR("[%s] syncMemSize is 0", __func__), HCCL_E_PARA);

    /* 4. 申请 syncMem 本地内存，后续基于 channel 交换。失败时回滚已创建的 team。 */
    void* syncMemPtr = nullptr;
    CHK_RET(AllocTeamSyncMem(worldTeam, syncMemSize, syncMemPtr));

    /* 5. 注册 worldTeam 到 HcclTeamMgr（存 syncMem + collComm 反查 + rankIds），供 HcclSubTeamCreate/HcclTeamDestroy
     * 反查。 */
    HcclResult regRet = HcclTeamMgr::GetInstance().RegisterWorldTeam(
        *worldTeam, collComm, syncMemPtr, syncMemSize, desc->rankIds, desc->rankNum);
    if (regRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] RegisterWorldTeam failed, comm[%s] ret[%d]", __func__, commId.c_str(), regRet);
        (void)hrtFree(syncMemPtr);
        (void)HcommTeamDestroy(*worldTeam);
        *worldTeam = nullptr;
        return regRet;
    }

    HCCL_INFO(
        "[%s] success, team[%p] comm[%s] rankNum[%u] selfRankId[%u] selfMemberId[%u] syncMemSize[%llu]", __func__,
        *worldTeam, commId.c_str(), desc->rankNum, desc->selfRankId, selfMemberId, syncMemSize);
    return HCCL_SUCCESS;
}

HcclResult HcclSubTeamCreate(HcommTeamHandle worldTeam, const HcclTeamCreateDesc* desc, HcommTeamHandle* team)
{
    CHK_PTR_NULL(worldTeam);
    CHK_PTR_NULL(desc);
    CHK_PTR_NULL(team);

    CHK_PRT_RET(
        (desc->rankNum == 0 || desc->rankNum == 1),
        HCCL_ERROR("[%s] sub team can not be empty or single rank", __func__), HCCL_E_PARA);
    CHK_PRT_RET(desc->rankIds == nullptr, HCCL_ERROR("[%s] rankIds is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        (desc->requirement.signalCount != 0 || desc->requirement.counterCount != 0),
        HCCL_ERROR(
            "[%s] signalCount[%u] and counterCount[%u] must be 0", __func__, desc->requirement.signalCount,
            desc->requirement.counterCount),
        HCCL_E_PARA);
    CHK_PRT_RET(
        desc->requirement.barrierCount == 0, HCCL_ERROR("[%s] barrierCount must be >= 1", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        HcclTeamMgr::GetInstance().FindWorldTeam(worldTeam) != worldTeam,
        HCCL_ERROR("[%s] team[%p] is not world team", __func__, worldTeam), HCCL_E_PARA);

    /* selfRankId 是本 rank 的实际 rankId，需在 rankIds 中查找其下标作为 memberId。 */
    uint32_t selfMemberId = 0;
    CHK_RET(FindSelfMemberId(desc, selfMemberId));

    /* 1. 反查 subTeam 的 rankIds 在 worldTeam.rankIds 中的下标作为 worldMemberIds（worldTeam 的 memberId）。 */
    std::vector<uint32_t> worldMemberIds;
    CHK_RET(BuildSubTeamWorldMemberIds(worldTeam, desc->rankIds, desc->rankNum, worldMemberIds));

    /* 2. 填充 HcommTeamCreateDesc：成员为 desc 的子集，父 team 为 worldTeam，worldMemberIds 用反查结果。 */
    HcommTeamCreateDesc hcommDesc = {};
    FillHcommTeamCreateDesc(desc, selfMemberId, worldMemberIds.data(), hcommDesc);

    /* 3. 调用 Hcomm 层创建 sub team，outSyncMemSize 为需要本地申请的 syncMem 字节数。 */
    uint64_t syncMemSize = 0;
    HcommResult ret = HcommTeamCreate(worldTeam, &hcommDesc, team, &syncMemSize);
    CHK_PRT_RET(
        ret != 0 || *team == nullptr,
        HCCL_ERROR(
            "[%s] HcommTeamCreate failed, ret[%d] rankNum[%u] selfRankId[%u]", __func__, ret, desc->rankNum,
            desc->selfRankId),
        (ret != 0) ? static_cast<HcclResult>(ret) : HCCL_E_INTERNAL);
    CHK_PRT_RET(syncMemSize == 0, HCCL_ERROR("[%s] syncMemSize is 0", __func__), HCCL_E_PARA);

    /* 4. 申请 syncMem 本地内存。失败时回滚已创建的 sub team。 */
    void* syncMemPtr = nullptr;
    CHK_RET(AllocTeamSyncMem(team, syncMemSize, syncMemPtr));

    /* 5. 注册 sub team 到 HcclTeamMgr（建父子关系 + 存 syncMem + rankIds，collComm 取自 worldTeam 条目）。 */
    HcclResult regRet = HcclTeamMgr::GetInstance().RegisterSubTeam(
        worldTeam, *team, syncMemPtr, syncMemSize, desc->rankIds, desc->rankNum);
    if (regRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] RegisterSubTeam failed, ret[%d]", __func__, regRet);
        (void)hrtFree(syncMemPtr);
        (void)HcommTeamDestroy(*team);
        *team = nullptr;
        return regRet;
    }

    HCCL_INFO(
        "[%s] success, team[%p] worldTeam[%p] rankNum[%u] selfRankId[%u] selfMemberId[%u] syncMemSize[%llu]", __func__,
        *team, worldTeam, desc->rankNum, desc->selfRankId, selfMemberId, syncMemSize);
    return HCCL_SUCCESS;
}

HcclResult HcclTeamDestroy(HcommTeamHandle team)
{
    CHK_PTR_NULL(team);

    /* worldTeam 销毁时连带销毁其所有 subTeam 与 window（1:N），避免泄漏。subTeam 无子 team、无 window。 */
    if (HcclTeamMgr::GetInstance().FindWorldTeam(team) == team) {
        // 先销毁所有 subTeam（递归调 HcclTeamDestroy，subTeam 不会重复进入此分支）
        std::vector<HcommTeamHandle> subTeams = HcclTeamMgr::GetInstance().GetSubTeams(team);
        for (HcommTeamHandle sub : subTeams) {
            (void)HcclTeamDestroy(sub);
        }
        // 再销毁 worldTeam 拥有的 window
        std::vector<WindowInfo> windows = HcclTeamMgr::GetInstance().GetWorldTeamWindows(team);
        for (const auto& win : windows) {
            if (win.handle != nullptr) {
                (void)HcommTeamWindowDeregister(team, win.handle);
            }
        }
    }

    /* 释放该 team 的 syncMem（hrtFree）+ erase 条目。 */
    HcclTeamMgr::GetInstance().UnregisterTeam(team);

    HcommResult ret = HcommTeamDestroy(team);
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("[%s] HcommTeamDestroy failed, ret[%d]", __func__, ret), static_cast<HcclResult>(ret));

    HCCL_INFO("[%s] success", __func__);
    return HCCL_SUCCESS;
}

/* 注册 team 的 syncMem 内存（team 粒度，仅首次注册一次）。
 * syncMemHandle 已存在则跳过注册；syncMemHandle 出参返回当前句柄（首次或已存在），供调用方日志使用。 */
static HcclResult
RegisterTeamSyncMem(HcommTeamHandle team, const std::string& commId, CommMems* commMem, HcclMemHandle& syncMemHandle)
{
    syncMemHandle = HcclTeamMgr::GetInstance().GetTeamSyncMemHandle(team);
    if (syncMemHandle != nullptr) {
        HCCL_INFO(
            "[RegisterTeamSyncMem] syncMemHandle[%p] already registered, comm[%s] team[%p]", syncMemHandle,
            commId.c_str(), team);
        return HCCL_SUCCESS;
    }
    void* syncMemPtr = HcclTeamMgr::GetInstance().GetSyncMemPtr(team);
    uint64_t syncMemSize = HcclTeamMgr::GetInstance().GetSyncMemSize(team);
    if (syncMemPtr == nullptr || syncMemSize == 0) {
        HCCL_ERROR(
            "[RegisterTeamSyncMem] syncMemPtr[%p] or syncMemSize[%llu] is invalid, comm[%s] team[%p]", syncMemPtr,
            syncMemSize, commId.c_str(), team);
        return HCCL_E_INTERNAL;
    }
    CommMem syncMemVar{};
    syncMemVar.type = COMM_MEM_TYPE_DEVICE;
    syncMemVar.addr = syncMemPtr;
    syncMemVar.size = syncMemSize;
    std::ostringstream syncMemTagStream;
    syncMemTagStream << HCCL_TEAM_SYNCMEM_TAG_PREFIX << commId << "_team_" << team << "_addr_" << syncMemPtr << "_size_"
                     << syncMemSize;
    const std::string syncMemTagStr = syncMemTagStream.str();
    HcclResult ret = commMem->CommRegMem(syncMemTagStr, syncMemVar, &syncMemHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[RegisterTeamSyncMem] register syncMem failed, comm[%s] ret[%d] size[%llu]", commId.c_str(), ret,
            syncMemSize),
        ret);
    HcclTeamMgr::GetInstance().SetTeamSyncMemHandle(team, syncMemHandle, syncMemTagStr);
    return HCCL_SUCCESS;
}

/* 重复注册检测：worldTeam 已注册过 window 且本次 localMem 是某 window 的 registeredLocalMem 的子集，则复用。
 * 命中复用时设 *window 并返回 true；否则返回 false。 */
static bool TryReuseWindow(
    HcommTeamHandle worldTeam, const CommMem& localMem, const std::string& commId, HcommTeamHandle team,
    HcommWindowHandle* window)
{
    HcommWindowHandle reusableWindow = nullptr;
    if (!HcclTeamMgr::GetInstance().FindReusableWindow(worldTeam, localMem, reusableWindow)) {
        return false;
    }
    *window = reusableWindow;
    HCCL_INFO(
        "[TryReuseWindow] reuse registered window, comm[%s] team[%p] worldTeam[%p] window[%p]", commId.c_str(), team,
        worldTeam, *window);
    return true;
}

/* 非复用路径：在 worldTeam 上创建业务 window，注册 localMem（失败回滚 window），记录到 worldTeam 的 window 列表。 */
static HcclResult CreateNewWindow(
    HcommTeamHandle worldTeam, const CommMem& localMem, const std::string& commId, CommMems* commMem,
    HcommWindowHandle* window)
{
    HcommResult hRet = HcommTeamWindowRegister(worldTeam, nullptr, window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC);
    if (hRet != 0 || *window == nullptr) {
        HCCL_ERROR(
            "[CreateNewWindow] HcommTeamWindowRegister failed, comm[%s] worldTeam[%p] ret[%d]", commId.c_str(),
            worldTeam, hRet);
        *window = nullptr;
        return (hRet != 0) ? static_cast<HcclResult>(hRet) : HCCL_E_INTERNAL;
    }
    std::ostringstream userTagStream;
    userTagStream << HCCL_TEAM_USERMEM_TAG_PREFIX << commId << "_addr_" << localMem.addr << "_size_" << localMem.size;
    const std::string userTag = userTagStream.str();
    HcclMemHandle userHandle = nullptr;
    HcclResult ret = commMem->CommRegMem(userTag, localMem, &userHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateNewWindow] register localMem failed, comm[%s] ret[%d]", commId.c_str(), ret);
        (void)HcommTeamWindowDeregister(worldTeam, *window);
        *window = nullptr;
        return ret;
    }
    HcclTeamMgr::GetInstance().AddWorldTeamWindow(worldTeam, *window, localMem, userHandle, userTag);
    return HCCL_SUCCESS;
}

HcclResult HcclTeamWindowRegister(
    HcclComm comm, HcommTeamHandle worldTeam, const CommMem* localMem, HcommWindowHandle* window, uint32_t flag)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(worldTeam);
    CHK_PTR_NULL(localMem);
    CHK_PTR_NULL(window);
    CHK_PRT_RET(flag != 0, HCCL_ERROR("[%s] flag[%u] is not supported, only support 0", __func__, flag), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem->type != COMM_MEM_TYPE_DEVICE),
        HCCL_ERROR("[%s] localMem type[%d] must be device", __func__, localMem->type), HCCL_E_PARA);
    CHK_PRT_RET(localMem->addr == nullptr, HCCL_ERROR("[%s] localMem addr is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(localMem->size == 0, HCCL_ERROR("[%s] localMem size is 0", __func__), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    const std::string commId = collComm->GetCommId();

    /* 入参 worldTeam 必须是 worldTeam（syncMem 注册已移至 ChannelsCreate，window 归 worldTeam 所有）。 */
    CHK_PRT_RET(
        HcclTeamMgr::GetInstance().FindWorldTeam(worldTeam) != worldTeam,
        HCCL_ERROR("[%s] worldTeam[%p] is not worldTeam", __func__, worldTeam), HCCL_E_PARA);
    CollComm* worldCollComm = HcclTeamMgr::GetInstance().FindCollComm(worldTeam);
    CHK_PTR_NULL(worldCollComm);
    CHK_PRT_RET(
        worldCollComm->GetCommId() != commId,
        HCCL_ERROR("[%s] worldTeam[%p] does not belong to comm[%s]", __func__, worldTeam, commId.c_str()), HCCL_E_PARA);

    auto myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    CommMems* commMem = myRank->GetCommMems();
    CHK_PTR_NULL(commMem);

    if (TryReuseWindow(worldTeam, *localMem, commId, worldTeam, window)) {
        return HCCL_SUCCESS;
    }

    CHK_RET(CreateNewWindow(worldTeam, *localMem, commId, commMem, window));

    HCCL_INFO("[%s] success, comm[%s] worldTeam[%p] window[%p]", __func__, commId.c_str(), worldTeam, *window);
    return HCCL_SUCCESS;
}

HcclResult HcclTeamWindowDeregister(HcommTeamHandle team, HcommWindowHandle window)
{
    CHK_PTR_NULL(team);
    CHK_PTR_NULL(window);

    /* 1. 取入参 team 对应的 worldTeam。 */
    HcommTeamHandle worldTeam = HcclTeamMgr::GetInstance().FindWorldTeam(team);
    CHK_PRT_RET(
        worldTeam == nullptr,
        HCCL_WARNING("[%s] FindWorldTeam failed, team[%p] not registered, maybe already destroyed", __func__, team),
        HCCL_SUCCESS);

    /* 2. 从 worldTeam 的 window 列表移除该 window 的记录（WindowInfo）。
     *    注：不注销 localMem 的 MemReg（由通信域析构兜底清理）；不处理 syncMem（team 粒度，team 销毁时释放）。 */
    HcclTeamMgr::GetInstance().RemoveWorldTeamWindow(worldTeam, window);

    /* 3. 销毁 Hcomm 层 window。 */
    HcommResult ret = HcommTeamWindowDeregister(worldTeam, window);
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("[%s] HcommTeamWindowDeregister failed, ret[%d]", __func__, ret),
        static_cast<HcclResult>(ret));

    HCCL_INFO("[%s] success, team[%p] worldTeam[%p] window[%p]", __func__, team, worldTeam, window);
    return HCCL_SUCCESS;
}

/* 查询本rank到peerRank的link，填充 HcclChannelDesc 的 endpoint/protocol 字段。*/
static HcclResult
FillChannelDescForPeer(HcclComm comm, HcommTeamHandle team, uint32_t selfRank, uint32_t peerRank, HcclChannelDesc& desc)
{
    uint32_t netLayer = 0;
    HcommResult hRet = HcommTeamGetNetLayer(team, &netLayer);
    CHK_PRT_RET(
        hRet != 0, HCCL_ERROR("[%s] HcommTeamGetNetLayer failed, ret[%d]", __func__, hRet),
        (hRet != 0) ? static_cast<HcclResult>(hRet) : HCCL_E_INTERNAL);

    CommLink* links = nullptr;
    uint32_t linkNum = 0;
    HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, selfRank, peerRank, &links, &linkNum);
    if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
        HCCL_ERROR("[%s] no link found from rank[%u] to rank[%u]", __func__, selfRank, peerRank);
        return HCCL_E_NOT_FOUND;
    }
    const CommLink& link = links[0];
    desc.remoteRank = peerRank;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    return HCCL_SUCCESS;
}

/* 从 HcclTeamMgr 取 team 的 rankIds（memberId→rankId），推算 memberNum 与 selfMemberId。 */
static HcclResult GetTeamMemberInfo(HcommTeamHandle team, uint32_t selfRank, ChannelsCreateCtx& ctx)
{
    ctx.rankIds = HcclTeamMgr::GetInstance().GetRankIds(team);
    CHK_PRT_RET(
        ctx.rankIds.empty(), HCCL_ERROR("[%s] GetRankIds failed, team[%p] not registered", __func__, team),
        HCCL_E_PARA);
    ctx.memberNum = static_cast<uint32_t>(ctx.rankIds.size());
    bool selfFound = false;
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        if (ctx.rankIds[m] == selfRank) {
            ctx.selfMemberId = m;
            selfFound = true;
            break;
        }
    }
    CHK_PRT_RET(!selfFound, HCCL_ERROR("[%s] selfRank[%u] not in team rankIds", __func__, selfRank), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

/* 取 worldTeam 及其所有 window、memHandles；取 worldTeam 维度信息并计算 curToWorld 映射。 */
static HcclResult GetWorldTeamContext(HcommTeamHandle team, uint32_t selfRank, ChannelsCreateCtx& ctx)
{
    ctx.worldTeam = HcclTeamMgr::GetInstance().FindWorldTeam(team);
    CHK_PTR_NULL(ctx.worldTeam);
    ctx.windows = HcclTeamMgr::GetInstance().GetWorldTeamWindows(ctx.worldTeam); // 移动赋值，避免深拷贝
    ctx.syncMemTag = HcclTeamMgr::GetInstance().GetTeamSyncMemTag(team);
    // 只收集未交换的 memHandles（syncMemHandle + window localMemHandle），避免重复建链交换
    ctx.memHandles = HcclTeamMgr::GetInstance().CollectPendingMemHandles(ctx.worldTeam, team);

    ctx.worldRankIds = HcclTeamMgr::GetInstance().GetRankIds(ctx.worldTeam);
    CHK_PRT_RET(
        ctx.worldRankIds.empty(),
        HCCL_ERROR("[%s] GetRankIds failed, worldTeam[%p] not registered", __func__, ctx.worldTeam), HCCL_E_PARA);
    ctx.worldMemberNum = static_cast<uint32_t>(ctx.worldRankIds.size());
    bool worldSelfFound = false;
    for (uint32_t w = 0; w < ctx.worldMemberNum; w++) {
        if (ctx.worldRankIds[w] == selfRank) {
            ctx.worldSelfMemberId = w;
            worldSelfFound = true;
            break;
        }
    }
    CHK_PRT_RET(
        !worldSelfFound, HCCL_ERROR("[%s] selfRank[%u] not in worldTeam rankIds", __func__, selfRank), HCCL_E_PARA);
    // curToWorld[m] = 当前 team memberId m 的 rankId 在 worldRankIds 中的下标（worldMemberId）
    ctx.curToWorld.assign(ctx.memberNum, 0);
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        uint32_t rankId = ctx.rankIds[m];
        for (uint32_t w = 0; w < ctx.worldMemberNum; w++) {
            if (ctx.worldRankIds[w] == rankId) {
                ctx.curToWorld[m] = w;
                break;
            }
        }
    }
    return HCCL_SUCCESS;
}

/* 对每个 peer member 创建 channelCnt 个 channel，结果存 ctx.channelsByMember（self 为空）。 */
static HcclResult AcquireChannels(
    HcclComm comm, HcommTeamHandle team, const HcclTeamCreateChannelsDesc* desc, uint32_t selfRank,
    ChannelsCreateCtx& ctx)
{
    uint32_t channelCnt = desc->channelCnt;
    ctx.channelsByMember.assign(ctx.memberNum, {});
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        if (m == ctx.selfMemberId) {
            continue; // 本 member 不建到自己的 channel
        }
        uint32_t peerRank = ctx.rankIds[m];
        std::vector<HcclChannelDesc> channelDescs(channelCnt);
        HcclResult ret = HcclChannelDescInit(channelDescs.data(), channelCnt);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclChannelDescInit failed, ret[%d]", __func__, ret), ret);
        for (uint32_t c = 0; c < channelCnt; c++) {
            ret = FillChannelDescForPeer(comm, team, selfRank, peerRank, channelDescs[c]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[%s] FillChannelDescForPeer failed, peerRank[%u] ret[%d]", __func__, peerRank, ret), ret);
            channelDescs[c].notifyNum = desc->notifyNum;
            if (!ctx.memHandles.empty()) {
                channelDescs[c].memHandles = ctx.memHandles.data();
                channelDescs[c].memHandleNum = static_cast<uint32_t>(ctx.memHandles.size());
            }
        }
        ctx.channelsByMember[m].assign(channelCnt, 0);
        ret = HcclChannelAcquire(comm, desc->engine, channelDescs.data(), channelCnt, ctx.channelsByMember[m].data());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] HcclChannelAcquire failed, peerMember[%u] peerRank[%u] ret[%d]", __func__, m, peerRank, ret),
            ret);
    }
    return HCCL_SUCCESS;
}

/* 构造 HcommTeamBindChannelsDesc，绑定 team 与 channel。 */
static HcclResult BindTeamChannels(HcommTeamHandle team, const std::string& commId, ChannelsCreateCtx& ctx)
{
    std::vector<uint32_t> channelNumPerMember(ctx.memberNum, 0);
    std::vector<uint64_t*> channelsByMemberIdPtrs(ctx.memberNum, nullptr);
    // 保持各 member 的 channel 数组生命周期覆盖 BindChannels 调用
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        channelNumPerMember[m] = static_cast<uint32_t>(ctx.channelsByMember[m].size());
        channelsByMemberIdPtrs[m] = ctx.channelsByMember[m].data();
    }
    HcommTeamBindChannelsDesc bindDesc = {};
    (void)HcommTeamBindChannelsDescInit(&bindDesc);
    bindDesc.memberNum = ctx.memberNum;
    bindDesc.channelNumPerMember = channelNumPerMember.data();
    bindDesc.channelsByMemberId = channelsByMemberIdPtrs.data();
    HcommResult hRet = HcommTeamBindChannels(team, &bindDesc);
    CHK_PRT_RET(
        hRet != 0,
        HCCL_ERROR(
            "[%s] HcommTeamBindChannels failed, comm[%s] ret[%d] memberNum[%u]", __func__, commId.c_str(), hRet,
            ctx.memberNum),
        static_cast<HcclResult>(hRet));
    return HCCL_SUCCESS;
}

/* 收集所有 peer channel 的远端内存+tag。selfMemberId 槽填本地 syncMem，peer 槽按 memberId 填远端 syncMem；
 * 各 window 的 localMem 远端按 (windowIndex, worldMemberId) 存 remoteMemsByWindow。 */
static HcclResult CollectRemoteMems(HcclComm comm, HcommTeamHandle team, ChannelsCreateCtx& ctx)
{
    ctx.syncMemRemoteMems.assign(ctx.memberNum, CommMem{});
    // selfMemberId 槽填本地 syncMem 内存
    void* localSyncMemPtr = HcclTeamMgr::GetInstance().GetSyncMemPtr(team);
    uint64_t localSyncMemSize = HcclTeamMgr::GetInstance().GetSyncMemSize(team);
    if (localSyncMemPtr != nullptr && localSyncMemSize > 0) {
        ctx.syncMemRemoteMems[ctx.selfMemberId].type = COMM_MEM_TYPE_DEVICE;
        ctx.syncMemRemoteMems[ctx.selfMemberId].addr = localSyncMemPtr;
        ctx.syncMemRemoteMems[ctx.selfMemberId].size = localSyncMemSize;
    }
    // window.mems 维度为 worldTeam memberNum，peer 槽按 worldMemberId 索引
    ctx.remoteMemsByWindow.assign(ctx.windows.size(), std::vector<CommMem>(ctx.worldMemberNum));
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        if (m == ctx.selfMemberId || ctx.channelsByMember[m].empty()) {
            continue;
        }
        uint32_t peerWorldMemberId = ctx.curToWorld[m];
        for (ChannelHandle channel : ctx.channelsByMember[m]) {
            if (channel == 0) {
                continue;
            }
            uint32_t memNum = 0;
            CommMem* remoteMems = nullptr;
            char** memTags = nullptr;
            HcclResult gRet = HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMems, &memTags);
            CHK_PRT_RET(
                gRet != HCCL_SUCCESS,
                HCCL_ERROR("[%s] HcclChannelGetRemoteMems failed, member[%u] ret[%d]", __func__, m, gRet), gRet);
            uint32_t userMemFilled = 0;
            for (uint32_t r = 0; r < memNum; r++) {
                std::string tag
                    = (memTags != nullptr && memTags[r] != nullptr) ? std::string(memTags[r]) : std::string();
                if (!ctx.syncMemTag.empty()
                    && tag.compare(0, strlen(HCCL_TEAM_SYNCMEM_TAG_PREFIX), HCCL_TEAM_SYNCMEM_TAG_PREFIX) == 0) {
                    ctx.syncMemRemoteMems[m] = remoteMems[r]; // 按下标 memberId 存
                    continue;
                }
                if (userMemFilled < ctx.windows.size()
                    && tag.compare(0, strlen(HCCL_TEAM_USERMEM_TAG_PREFIX), HCCL_TEAM_USERMEM_TAG_PREFIX) == 0) {
                    ctx.remoteMemsByWindow[userMemFilled][peerWorldMemberId] = remoteMems[r];
                    userMemFilled++;
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

/* per window 调 HcommTeamWindowBindRemoteMems（worldMemberId 维 mems）；调 HcommTeamBindRemoteSyncMem 绑定 syncMem。 */
static HcclResult BindWindowsAndSyncMem(HcommTeamHandle team, const std::string& commId, ChannelsCreateCtx& ctx)
{
    // 对每个 window 调 HcommTeamWindowBindRemoteMems
    for (size_t w = 0; w < ctx.windows.size(); w++) {
        std::vector<CommMem> windowMems(ctx.worldMemberNum);
        for (uint32_t m = 0; m < ctx.worldMemberNum; m++) {
            if (m == ctx.worldSelfMemberId) {
                continue;
            }
            windowMems[m] = ctx.remoteMemsByWindow[w][m];
        }
        // self 槽填调用者在 WindowRegister 时传入的本地内存（registeredLocalMem），与 peer 槽的远端 CommMem 对称
        windowMems[ctx.worldSelfMemberId] = ctx.windows[w].registeredLocalMem;
        HcommTeamWindowDesc winDesc = {};
        (void)HcommTeamWindowDescInit(&winDesc);
        winDesc.mems = windowMems.data();
        winDesc.memberNum = ctx.worldMemberNum;
        HcommResult bindWinRet = HcommTeamWindowBindRemoteMems(team, ctx.windows[w].handle, &winDesc);
        CHK_PRT_RET(
            bindWinRet != 0,
            HCCL_ERROR(
                "[%s] HcommTeamWindowBindRemoteMems failed, comm[%s] team[%p] window[%p] ret[%d]", __func__,
                commId.c_str(), team, ctx.windows[w].handle, bindWinRet),
            static_cast<HcclResult>(bindWinRet));
    }
    // 调 HcommTeamBindRemoteSyncMem
    HcommTeamBindSyncMemDesc syncDesc = {};
    (void)HcommTeamBindSyncMemDescInit(&syncDesc);
    syncDesc.remoteMems = ctx.syncMemRemoteMems.data();
    syncDesc.remoteMemNum = static_cast<uint32_t>(ctx.syncMemRemoteMems.size());
    HcommResult bindSyncRet = HcommTeamBindRemoteSyncMem(team, &syncDesc);
    CHK_PRT_RET(
        bindSyncRet != 0,
        HCCL_ERROR(
            "[%s] HcommTeamBindRemoteSyncMem failed, comm[%s] team[%p] ret[%d] remoteMemNum[%zu]", __func__,
            commId.c_str(), team, bindSyncRet, ctx.syncMemRemoteMems.size()),
        static_cast<HcclResult>(bindSyncRet));
    return HCCL_SUCCESS;
}

HcclResult HcclTeamChannelsCreate(HcclComm comm, HcommTeamHandle team, const HcclTeamCreateChannelsDesc* desc)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(team);
    CHK_PTR_NULL(desc);
    CHK_PRT_RET(desc->channelCnt == 0, HCCL_ERROR("[%s] channelCnt is 0", __func__), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    const std::string commId = collComm->GetCommId();
    uint32_t selfRank = collComm->GetMyRankId();

    CollComm* worldCollComm = HcclTeamMgr::GetInstance().FindCollComm(team);
    CHK_PTR_NULL(worldCollComm);
    CHK_PRT_RET(
        worldCollComm->GetCommId() != commId,
        HCCL_ERROR("[%s] team[%p] does not belong to comm[%s]", __func__, team, commId.c_str()), HCCL_E_PARA);

    ChannelsCreateCtx ctx{};
    CHK_RET(GetTeamMemberInfo(team, selfRank, ctx));

    /* 注册 team 的 syncMem 内存（team 粒度，仅首次注册一次），供 channel 交换与 GetWorldTeamContext 取用。 */
    auto myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    CommMems* commMem = myRank->GetCommMems();
    CHK_PTR_NULL(commMem);
    HcclMemHandle syncMemHandle = nullptr;
    CHK_RET(RegisterTeamSyncMem(team, commId, commMem, syncMemHandle));

    CHK_RET(GetWorldTeamContext(team, selfRank, ctx));
    CHK_RET(AcquireChannels(comm, team, desc, selfRank, ctx));
    CHK_RET(BindTeamChannels(team, commId, ctx));
    CHK_RET(CollectRemoteMems(comm, team, ctx));
    CHK_RET(BindWindowsAndSyncMem(team, commId, ctx));

    HCCL_INFO(
        "[%s] success, comm[%s] team[%p] memberNum[%u] channelCnt[%u] windowNum[%zu] syncMemRemoteMemNum[%zu]",
        __func__, commId.c_str(), team, ctx.memberNum, desc->channelCnt, ctx.windows.size(),
        ctx.syncMemRemoteMems.size());
    return HCCL_SUCCESS;
}
