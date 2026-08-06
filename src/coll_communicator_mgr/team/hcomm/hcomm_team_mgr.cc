/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_team_mgr.h"

#include <cstdlib>

#include "adapter_rts_common.h"
#include "hcomm_res_entity_defs.h"
#include "hcomm_team_entity_defs.h"
#include "log.h"
#include "securec.h"

namespace hcomm {

HcommTeamMgr &HcommTeamMgr::GetInstance()
{
    static std::once_flag instanceFlag;
    static HcommTeamMgr *instance = nullptr;
    std::call_once(instanceFlag, [&] { instance = new HcommTeamMgr(); });
    return *instance;
}

HcommTeamMgr::~HcommTeamMgr()
{
    {
        std::unique_lock<std::shared_mutex> lock(teamsRwMutex_);
        for (auto &pair : teams_) {
            if (pair.second != nullptr) {
                FreeTeamResources(pair.second.get());
            }
        }
        teams_.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(windowsRwMutex_);
        for (auto &pair : windows_) {
            if (pair.second != nullptr) {
                FreeWindowResources(pair.second.get());
            }
        }
        windows_.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(windowToTeamRwMutex_);
        windowToTeamMap_.clear();
    }
}

TeamEntry *HcommTeamMgr::FindTeamByHandleLocked(HcommTeamHandle handle)
{
    auto it = teams_.find(handle);
    if (it == teams_.end()) {
        HCCL_ERROR("[FindTeamByHandleLocked] team handle[%p] not found", handle);
        return nullptr;
    }
    return it->second.get();
}

WindowEntry *HcommTeamMgr::FindWindowByHandleLocked(HcommWindowHandle handle)
{
    auto it = windows_.find(handle);
    if (it == windows_.end()) {
        HCCL_ERROR("[FindWindowByHandleLocked] window handle[%p] not found", handle);
        return nullptr;
    }
    return it->second.get();
}

HcommResult HcommTeamMgr::SyncTeamToDevice(TeamEntry *entry)
{
    if (entry->devTeam == nullptr) {
        HCCL_ERROR("[SyncTeamToDevice] devTeam is null");
        return HCOMM_E_PTR;
    }
    return static_cast<HcommResult>(
        hrtMemSyncCopy(entry->devTeam, sizeof(HcommTeam), &entry->hostTeam, sizeof(HcommTeam),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
}

HcommResult HcommTeamMgr::SyncWindowToDevice(WindowEntry *entry)
{
    if (entry->devWindow == nullptr) {
        HCCL_ERROR("[SyncWindowToDevice] devWindow is null");
        return HCOMM_E_PTR;
    }
    return static_cast<HcommResult>(
        hrtMemSyncCopy(entry->devWindow, sizeof(HcommWindow), &entry->hostWindow, sizeof(HcommWindow),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
}

HcommResult HcommTeamMgr::AllocAndCopyWorldTeamIds(TeamEntry *entry, const uint32_t *src,
                                                     uint32_t memberNum)
{
    // worldTeam（src==nullptr）：worldTeamIds 置为 [0,memberNum) 连续序列（memberId 自身）
    std::vector<uint32_t> sequentialIds;
    const uint32_t *srcPtr = src;
    if (src == nullptr) {
        sequentialIds.resize(memberNum);
        for (uint32_t i = 0; i < memberNum; i++) {
            sequentialIds[i] = i;
        }
        srcPtr = sequentialIds.data();
    }

    void *devPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(
        hrtMalloc(&devPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t))));
    CHK_PRT_RET(ret != HCOMM_SUCCESS,
        HCCL_ERROR("[AllocAndCopyWorldTeamIds] hrtMalloc failed, ret[%d]", ret), ret);

    ret = static_cast<HcommResult>(hrtMemSyncCopy(devPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t)),
        srcPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t)),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndCopyWorldTeamIds] hrtMemSyncCopy failed, ret[%d]", ret);
        (void)hrtFree(devPtr);
        return ret;
    }

    entry->devWorldTeamIds = devPtr;
    entry->hostTeam.worldTeamIds = static_cast<uint32_t *>(devPtr);

    entry->hostWorldTeamIds = static_cast<uint32_t *>(malloc(memberNum * sizeof(uint32_t)));
    if (entry->hostWorldTeamIds == nullptr) {
        HCCL_ERROR("[AllocAndCopyWorldTeamIds] malloc hostWorldTeamIds failed");
        (void)hrtFree(devPtr);
        entry->devWorldTeamIds = nullptr;
        entry->hostTeam.worldTeamIds = nullptr;
        return HCOMM_E_MEMORY;
    }
    errno_t memRet = memcpy_s(entry->hostWorldTeamIds, memberNum * sizeof(uint32_t),
        srcPtr, memberNum * sizeof(uint32_t));
    if (memRet != EOK) {
        HCCL_ERROR("[AllocAndCopyWorldTeamIds] memcpy_s hostWorldTeamIds failed, ret[%d]", memRet);
        free(entry->hostWorldTeamIds);
        entry->hostWorldTeamIds = nullptr;
        (void)hrtFree(devPtr);
        entry->devWorldTeamIds = nullptr;
        entry->hostTeam.worldTeamIds = nullptr;
        return HCOMM_E_MEMORY;
    }

    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocChannelEntities(TeamEntry *entry)
{
    uint32_t memberNum = entry->hostTeam.memberNum;
    entry->hostChannelNums.resize(memberNum, 0);
    uint32_t totalChannels = 0;
    for (uint32_t i = 0; i < memberNum; i++) {
        uint32_t chNum = static_cast<uint32_t>(entry->channelsList[i].size());
        entry->hostChannelNums[i] = chNum;
        totalChannels += chNum;
    }

    if (totalChannels == 0) {
        entry->devChannels = nullptr;
        entry->hostTeam.channelsBaseAddr = 0;
        return HCOMM_SUCCESS;
    }

    void *devPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(
        hrtMalloc(&devPtr, static_cast<uint64_t>(totalChannels * sizeof(ChannelEntity))));
    CHK_PRT_RET(ret != HCOMM_SUCCESS,
        HCCL_ERROR("[AllocChannelEntities] hrtMalloc failed, totalChannels[%u] ret[%d]", totalChannels, ret), ret);

    // 逐个 channel 做 D2D 拷贝：把 ChannelHandle 指向的 ChannelEntity 本体拷贝到连续内存对应偏移
    // 偏移按前缀和 sum(channelNumPerMember[0..peer-1]) + channelIdx 计算
    uint32_t offset = 0;
    for (uint32_t peer = 0; peer < memberNum; peer++) {
        for (uint32_t idx = 0; idx < entry->channelsList[peer].size(); idx++) {
            void *dst = reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(devPtr) + offset * sizeof(ChannelEntity));
            void *src = reinterpret_cast<void *>(static_cast<uintptr_t>(entry->channelsList[peer][idx]));
            ret = static_cast<HcommResult>(hrtMemSyncCopy(dst, sizeof(ChannelEntity), src, sizeof(ChannelEntity),
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE));
            if (ret != HCOMM_SUCCESS) {
                HCCL_ERROR("[AllocChannelEntities] hrtMemSyncCopy D2D failed, peer[%u] idx[%u] ret[%d]", peer, idx, ret);
                (void)hrtFree(devPtr);
                return ret;
            }
            offset++;
        }
    }

    entry->devChannels = devPtr;
    entry->hostTeam.channelsBaseAddr = reinterpret_cast<uint64_t>(devPtr);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocChannelNumsArray(TeamEntry *entry)
{
    uint32_t memberNum = entry->hostTeam.memberNum;
    void *devPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(
        hrtMalloc(&devPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t))));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocChannelNumsArray] hrtMalloc channelNums failed, ret[%d]", ret);
        return ret;
    }

    ret = static_cast<HcommResult>(hrtMemSyncCopy(devPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t)),
        entry->hostChannelNums.data(), static_cast<uint64_t>(memberNum * sizeof(uint32_t)),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocChannelNumsArray] hrtMemSyncCopy channelNums failed, ret[%d]", ret);
        (void)hrtFree(devPtr);
        return ret;
    }
    entry->devChannelNums = devPtr;
    entry->hostTeam.channelNumPerMember = static_cast<uint32_t *>(devPtr);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocAndCopyChannels(TeamEntry *entry)
{
    uint32_t memberNum = entry->hostTeam.memberNum;
    if (memberNum == 0 || entry->channelsList.empty()) {
        HCCL_ERROR("[AllocAndCopyChannels] memberNum[%u] or channelsList is empty", memberNum);
        return HCOMM_E_PARA;
    }

    FreeDeviceChannels(entry);

    HcommResult ret = AllocChannelEntities(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndCopyChannels] AllocChannelEntities failed, ret[%d]", ret);
        return ret;
    }

    ret = AllocChannelNumsArray(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndCopyChannels] AllocChannelNumsArray failed, ret[%d]", ret);
        FreeDeviceChannels(entry);
        return ret;
    }

    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocAndCopyRemoteMems(TeamEntry *entry, const CommMem *src,
                                                     uint32_t memberNum)
{
    if (entry->hostRemoteMems == nullptr) {
        return AllocRemoteMems(entry, src, memberNum);
    }
    return UpdateRemoteMems(entry, src, memberNum);
}

HcommResult HcommTeamMgr::AllocRemoteMems(TeamEntry *entry, const CommMem *src, uint32_t memberNum)
{
    // 首次分配：calloc hostRemoteMems + memcpy 拷入；hrtMalloc devRemoteMems + hrtMemSyncCopy 到 device
    entry->hostRemoteMems = static_cast<CommMem *>(calloc(memberNum, sizeof(CommMem)));
    if (entry->hostRemoteMems == nullptr) {
        HCCL_ERROR("[AllocRemoteMems] calloc hostRemoteMems failed");
        return HCOMM_E_MEMORY;
    }
    errno_t memRet = memcpy_s(entry->hostRemoteMems, memberNum * sizeof(CommMem),
        src, memberNum * sizeof(CommMem));
    if (memRet != EOK) {
        HCCL_ERROR("[AllocRemoteMems] memcpy_s failed, ret[%d]", memRet);
        free(entry->hostRemoteMems);
        entry->hostRemoteMems = nullptr;
        return HCOMM_E_MEMORY;
    }

    void *devPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(
        hrtMalloc(&devPtr, static_cast<uint64_t>(memberNum * sizeof(CommMem))));
    CHK_PRT_RET(ret != HCOMM_SUCCESS,
        HCCL_ERROR("[AllocRemoteMems] hrtMalloc failed, ret[%d]", ret), ret);

    ret = static_cast<HcommResult>(hrtMemSyncCopy(devPtr, static_cast<uint64_t>(memberNum * sizeof(CommMem)),
        src, static_cast<uint64_t>(memberNum * sizeof(CommMem)),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocRemoteMems] hrtMemSyncCopy failed, ret[%d]", ret);
        (void)hrtFree(devPtr);
        return ret;
    }

    entry->devRemoteMems = devPtr;
    entry->hostTeam.syncMem.remoteMems = static_cast<CommMem *>(devPtr);
    entry->hostTeam.syncMem.remoteMemsNum = memberNum;
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::UpdateRemoteMems(TeamEntry *entry, const CommMem *src, uint32_t memberNum)
{
    // 重复 bind：校验维度一致后更新 hostRemoteMems 并重新 sync 到 devRemoteMems
    CHK_PRT_RET(memberNum != entry->hostTeam.syncMem.remoteMemsNum,
        HCCL_ERROR("[UpdateRemoteMems] memberNum[%u] != remoteMemNum[%u], dimension mismatch",
            memberNum, entry->hostTeam.syncMem.remoteMemsNum),
        HCOMM_E_PARA);

    errno_t memRet = memcpy_s(entry->hostRemoteMems, memberNum * sizeof(CommMem),
        src, memberNum * sizeof(CommMem));
    CHK_PRT_RET(memRet != EOK,
        HCCL_ERROR("[UpdateRemoteMems] memcpy_s failed, ret[%d]", memRet), HCOMM_E_MEMORY);

    HcclResult hrtRet = hrtMemSyncCopy(entry->devRemoteMems,
        static_cast<uint64_t>(memberNum * sizeof(CommMem)),
        entry->hostRemoteMems, static_cast<uint64_t>(memberNum * sizeof(CommMem)),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    HcommResult ret = static_cast<HcommResult>(hrtRet);
    CHK_PRT_RET(ret != HCOMM_SUCCESS,
        HCCL_ERROR("[UpdateRemoteMems] hrtMemSyncCopy failed, ret[%d]", ret), ret);
    return HCOMM_SUCCESS;
}

void HcommTeamMgr::FreeDeviceChannels(TeamEntry *entry)
{
    if (entry->devChannels != nullptr) {
        (void)hrtFree(entry->devChannels);
        entry->devChannels = nullptr;
    }
    if (entry->devChannelNums != nullptr) {
        (void)hrtFree(entry->devChannelNums);
        entry->devChannelNums = nullptr;
    }
    entry->hostTeam.channelsBaseAddr = 0;
    entry->hostTeam.channelNumPerMember = nullptr;
}

void HcommTeamMgr::FreeTeamResources(TeamEntry *entry)
{
    if (entry->devRemoteMems != nullptr) {
        (void)hrtFree(entry->devRemoteMems);
        entry->devRemoteMems = nullptr;
    }
    if (entry->hostRemoteMems != nullptr) {
        free(entry->hostRemoteMems);
        entry->hostRemoteMems = nullptr;
    }

    FreeDeviceChannels(entry);

    if (entry->devWorldTeamIds != nullptr) {
        (void)hrtFree(entry->devWorldTeamIds);
        entry->devWorldTeamIds = nullptr;
    }
    if (entry->hostWorldTeamIds != nullptr) {
        free(entry->hostWorldTeamIds);
        entry->hostWorldTeamIds = nullptr;
    }

    if (entry->hostTeam.syncMem.shadowMem.addr != nullptr) {
        (void)hrtFree(entry->hostTeam.syncMem.shadowMem.addr);
        entry->hostTeam.syncMem.shadowMem.addr = nullptr;
        entry->hostTeam.syncMem.shadowMem.size = 0;
    }

    if (entry->devTeam != nullptr) {
        (void)hrtFree(entry->devTeam);
        entry->devTeam = nullptr;
    }

    entry->channelsList.clear();
    entry->hostChannelNums.clear();
}

void HcommTeamMgr::FreeWindowResources(WindowEntry *entry)
{
    if (entry->devMems != nullptr) {
        (void)hrtFree(entry->devMems);
        entry->devMems = nullptr;
    }
    if (entry->hostMems != nullptr) {
        free(entry->hostMems);
        entry->hostMems = nullptr;
    }
    if (entry->devWindow != nullptr) {
        (void)hrtFree(entry->devWindow);
        entry->devWindow = nullptr;
    }
}

HcommResult HcommTeamMgr::ValidateSubTeam(TeamEntry *worldEntry, const HcommTeamCreateDesc *desc)
{
    /* worldEntry 为 nullptr 表示创建 world team（无父 team），无需 sub team 成员校验。 */
    CHK_PRT_RET(worldEntry == nullptr, HCCL_INFO("[TeamCreate] worldEntry is null, skip sub team validation"),
        HCOMM_SUCCESS);
    CHK_PRT_RET(desc->memberNum > worldEntry->hostTeam.memberNum,
        HCCL_ERROR("[TeamCreate] sub team memberNum[%u] > world team memberNum[%u]",
            desc->memberNum, worldEntry->hostTeam.memberNum), HCOMM_E_PARA);

    uint32_t worldMemberNum = worldEntry->hostTeam.memberNum;
    for (uint32_t i = 0; i < desc->memberNum; i++) {
        CHK_PRT_RET(desc->worldMemberIds[i] >= worldMemberNum,
            HCCL_ERROR("[TeamCreate] worldMemberIds[%u]=%u >= world team memberNum[%u]",
                i, desc->worldMemberIds[i], worldMemberNum), HCOMM_E_PARA);
    }
    return HCOMM_SUCCESS;
}

void HcommTeamMgr::InitTeamEntry(TeamEntry *entry, const HcommTeamCreateDesc *desc,
                                    HcommTeamHandle worldTeam)
{
    entry->hostTeam.header.version = HCOMM_TEAM_VERSION;
    entry->hostTeam.header.magicWord = HCOMM_TEAM_MAGIC_WORD;
    entry->hostTeam.header.size = sizeof(HcommTeam);
    entry->hostTeam.header.reserved = 0;
    entry->hostTeam.memberNum = desc->memberNum;
    entry->hostTeam.selfMemberId = desc->selfMemberId;
    entry->hostTeam.netLayer = desc->netLayer;
    entry->hostTeam.engine = COMM_ENGINE_RESERVED;
    entry->syncMemReq = desc->requirement;
    entry->syncMemSize = static_cast<uint64_t>(desc->requirement.signalCount
        + desc->requirement.counterCount + desc->requirement.barrierCount)
        * sizeof(uint64_t) * desc->memberNum;
    entry->hostTeam.syncMem.syncMemReq = desc->requirement;
    entry->hostTeam.syncMem.syncMemSize = entry->syncMemSize;

    if (worldTeam != nullptr) {
        entry->worldTeamHandle = worldTeam;
        entry->isSubTeam = true;
    }
}

HcommResult HcommTeamMgr::AllocAndSyncTeam(TeamEntry *entry, const HcommTeamCreateDesc *desc)
{
    HcommResult ret = AllocAndCopyWorldTeamIds(entry, desc->worldMemberIds, desc->memberNum);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndSyncTeam] AllocAndCopyWorldTeamIds failed");
        FreeTeamResources(entry);
        return ret;
    }

    void *devTeamPtr = nullptr;
    ret = static_cast<HcommResult>(
        hrtMalloc(&devTeamPtr, static_cast<uint64_t>(sizeof(HcommTeam))));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndSyncTeam] hrtMalloc devTeam failed, ret[%d]", ret);
        FreeTeamResources(entry);
        return ret;
    }
    entry->devTeam = static_cast<HcommTeamHandle>(devTeamPtr);

    ret = SyncTeamToDevice(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndSyncTeam] SyncTeamToDevice failed, ret[%d]", ret);
        FreeTeamResources(entry);
        return ret;
    }
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::TeamCreate(HcommTeamHandle worldTeam, const HcommTeamCreateDesc *desc,
                                       HcommTeamHandle *team, uint64_t *outSyncMemSize)
{
    TeamEntry *worldEntry = nullptr;
    if (worldTeam != nullptr) {
        std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
        worldEntry = FindTeamByHandleLocked(worldTeam);
        CHK_PRT_RET(worldEntry == nullptr,
            HCCL_ERROR("[TeamCreate] worldTeam handle[%p] not found", worldTeam), HCOMM_E_NOT_FOUND);
    }

    HcommResult ret = ValidateSubTeam(worldEntry, desc);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[TeamCreate] ValidateSubTeam failed"), ret);

    auto entry = std::make_unique<TeamEntry>();
    InitTeamEntry(entry.get(), desc, worldTeam);

    ret = AllocAndSyncTeam(entry.get(), desc);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[TeamCreate] AllocAndSyncTeam failed"), ret);

    *outSyncMemSize = entry->syncMemSize;
    *team = entry->devTeam;

    {
        std::unique_lock<std::shared_mutex> lock(teamsRwMutex_);
        teams_[*team] = std::move(entry);
    }

    HCCL_INFO("[TeamCreate] team created, memberNum[%u], selfMemberId[%u], "
        "netLayer[%u], protocol[%d], "
        "signalCount[%u], counterCount[%u], barrierCount[%u], "
        "syncMemSize[%llu], isSubTeam[%d], devTeam[%p]",
        desc->memberNum, desc->selfMemberId, desc->netLayer, static_cast<int32_t>(desc->protocol),
        desc->requirement.signalCount, desc->requirement.counterCount,
        desc->requirement.barrierCount, *outSyncMemSize,
        static_cast<int32_t>(worldTeam != nullptr), *team);

    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::TeamDestroy(HcommTeamHandle team)
{
    std::unique_lock<std::shared_mutex> lock(teamsRwMutex_);
    auto it = teams_.find(team);
    CHK_PRT_RET(it == teams_.end(),
        HCCL_WARNING("[TeamDestroy] team handle[%p] not found, maybe already destroyed", team), HCOMM_SUCCESS);

    bool isSubTeam = it->second->isSubTeam;
    FreeTeamResources(it->second.get());
    teams_.erase(it);

    HCCL_INFO("[TeamDestroy] team[%p] destroyed, isSubTeam[%d]", team, static_cast<int32_t>(isSubTeam));
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::WindowRegister(HcommTeamHandle team, HcommWindowHandle *handle)
{
    // 校验 team 存在且为 worldTeam：shared_lock 仅覆盖查找/校验，不可延伸到 unique_lock 区间，否则自死锁。
    {
        std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
        TeamEntry *entry = FindTeamByHandleLocked(team);
        CHK_PRT_RET(entry == nullptr,
            HCCL_ERROR("[WindowRegister] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);
        CHK_PRT_RET(entry->isSubTeam,
            HCCL_ERROR("[WindowRegister] subteam cannot register window, use worldTeam"), HCOMM_E_PARA);
    }

    auto winEntry = std::make_unique<WindowEntry>();
    winEntry->teamHandle = team;
    winEntry->hostWindow.header.version = HCOMM_WINDOW_VERSION;
    winEntry->hostWindow.header.magicWord = HCOMM_WINDOW_MAGIC_WORD;
    winEntry->hostWindow.header.size = sizeof(HcommWindow);
    winEntry->hostWindow.header.reserved = 0;
    winEntry->hostWindow.worldTeam = team;

    void *devWindowPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(hrtMalloc(&devWindowPtr, static_cast<uint64_t>(sizeof(HcommWindow))));
    CHK_PRT_RET(ret != HCOMM_SUCCESS,
        HCCL_ERROR("[WindowRegister] hrtMalloc devWindow failed, ret[%d]", ret), ret);
    winEntry->devWindow = static_cast<HcommWindowHandle>(devWindowPtr);

    ret = SyncWindowToDevice(winEntry.get());
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[WindowRegister] SyncWindowToDevice failed, ret[%d]", ret);
        FreeWindowResources(winEntry.get());
        return ret;
    }

    *handle = winEntry->devWindow;

    {
        // 同时持有 windows_ 与 windowToTeamMap_ 写锁，保证 window 登记原子性（锁顺序：windows→windowToTeam）
        std::unique_lock<std::shared_mutex> winLock(windowsRwMutex_);
        std::unique_lock<std::shared_mutex> mapLock(windowToTeamRwMutex_);
        windowToTeamMap_[*handle] = team;
        windows_[*handle] = std::move(winEntry);
    }

    HCCL_INFO("[WindowRegister] window created, team[%p], devWindow[%p]", team, devWindowPtr);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocAndCopyWindowMems(WindowEntry *winEntry, uint64_t memberNum,
                                                    const CommMem *src)
{
    winEntry->hostMems = static_cast<CommMem *>(calloc(memberNum, sizeof(CommMem)));
    if (winEntry->hostMems == nullptr) {
        HCCL_ERROR("[AllocAndCopyWindowMems] calloc hostMems failed");
        return HCOMM_E_PTR;
    }
    errno_t memRet = memcpy_s(winEntry->hostMems, memberNum * sizeof(CommMem),
        src, memberNum * sizeof(CommMem));
    if (memRet != EOK) {
        HCCL_ERROR("[AllocAndCopyWindowMems] memcpy_s hostMems failed, ret[%d]", memRet);
        free(winEntry->hostMems);
        winEntry->hostMems = nullptr;
        return HCOMM_E_MEMORY;
    }

    void *devMemsPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(
        hrtMalloc(&devMemsPtr, static_cast<uint64_t>(memberNum * sizeof(CommMem))));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndCopyWindowMems] hrtMalloc devMems failed, ret[%d]", ret);
        free(winEntry->hostMems);
        winEntry->hostMems = nullptr;
        return ret;
    }

    ret = static_cast<HcommResult>(hrtMemSyncCopy(devMemsPtr, static_cast<uint64_t>(memberNum * sizeof(CommMem)),
        src, static_cast<uint64_t>(memberNum * sizeof(CommMem)),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndCopyWindowMems] hrtMemSyncCopy devMems failed, ret[%d]", ret);
        (void)hrtFree(devMemsPtr);
        free(winEntry->hostMems);
        winEntry->hostMems = nullptr;
        return ret;
    }
    winEntry->devMems = devMemsPtr;
    winEntry->hostWindow.mems = static_cast<CommMem *>(devMemsPtr);
    winEntry->hostWindow.memsNum = memberNum;
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::MergeWindowMems(WindowEntry *winEntry, uint64_t memberNum, const CommMem *src)
{
    CHK_PRT_RET(winEntry->hostMems == nullptr || winEntry->devMems == nullptr,
        HCCL_ERROR("[MergeWindowMems] hostMems or devMems is null, not allocated yet"), HCOMM_E_PTR);
    CHK_PRT_RET(memberNum != winEntry->hostWindow.memsNum,
        HCCL_ERROR("[MergeWindowMems] memberNum[%llu] != memsNum[%llu], dimension mismatch",
            memberNum, winEntry->hostWindow.memsNum), HCOMM_E_PARA);

    for (uint64_t i = 0; i < memberNum; i++) {
        if (src[i].addr != nullptr) {
            CHK_PRT_RET(winEntry->hostMems[i].addr != nullptr,
                HCCL_ERROR("[MergeWindowMems] member[%llu] already bound, register a new window to rebind", i),
                HCOMM_E_PARA);
            winEntry->hostMems[i] = src[i];
        }
    }

    // 重新把 hostMems 整块 sync 到 devMems
    HcclResult hrtRet = hrtMemSyncCopy(winEntry->devMems, static_cast<uint64_t>(memberNum * sizeof(CommMem)),
        winEntry->hostMems, static_cast<uint64_t>(memberNum * sizeof(CommMem)),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    HcommResult ret = static_cast<HcommResult>(hrtRet);
    CHK_PRT_RET(ret != HCOMM_SUCCESS,
        HCCL_ERROR("[MergeWindowMems] hrtMemSyncCopy devMems failed, ret[%d]", ret), ret);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::BindWindow(HcommTeamHandle team, HcommWindowHandle handle,
                                      const HcommTeamWindowDesc *desc)
{
    // 同时读 windows_ 与 teams_ 校验存在性（锁顺序：teams→windows）
    std::shared_lock<std::shared_mutex> teamLock(teamsRwMutex_);
    std::shared_lock<std::shared_mutex> winLock(windowsRwMutex_);
    WindowEntry *winEntry = FindWindowByHandleLocked(handle);
    CHK_PRT_RET(winEntry == nullptr,
        HCCL_ERROR("[BindWindow] window handle[%p] not found", handle), HCOMM_E_NOT_FOUND);
    TeamEntry *teamEntry = FindTeamByHandleLocked(team);
    CHK_PRT_RET(teamEntry == nullptr,
        HCCL_ERROR("[BindWindow] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);

    (void)teamEntry;
    HcommResult ret;
    if (winEntry->hostMems == nullptr) {
        ret = AllocAndCopyWindowMems(winEntry, desc->memberNum, desc->mems);
        CHK_PRT_RET(ret != HCOMM_SUCCESS,
            HCCL_ERROR("[BindWindow] AllocAndCopyWindowMems failed"), ret);
    } else {
        ret = MergeWindowMems(winEntry, desc->memberNum, desc->mems);
        CHK_PRT_RET(ret != HCOMM_SUCCESS,
            HCCL_ERROR("[BindWindow] MergeWindowMems failed"), ret);
    }

    ret = SyncWindowToDevice(winEntry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[BindWindow] SyncWindowToDevice failed, ret[%d]", ret);
        return ret;
    }

    HCCL_INFO("[BindWindow] window bound, team[%p], handle[%p], devWindow[%p], mems[%p], memberNum[%u], devMems[%p]",
        team, handle, winEntry->devWindow, desc->mems, desc->memberNum, winEntry->devMems);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::WindowDeregister(HcommTeamHandle team, HcommWindowHandle handle)
{
    {
        std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
        TeamEntry *entry = FindTeamByHandleLocked(team);
        CHK_PRT_RET(entry == nullptr,
            HCCL_ERROR("[WindowDeregister] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);
        CHK_PRT_RET(entry->isSubTeam,
            HCCL_ERROR("[WindowDeregister] subteam cannot deregister window, use worldTeam"), HCOMM_E_PARA);
    }
    // 同时持有 windows_ 与 windowToTeamMap_ 写锁，保证销毁原子性（锁顺序：windows→windowToTeam）
    std::unique_lock<std::shared_mutex> winLock(windowsRwMutex_);
    std::unique_lock<std::shared_mutex> mapLock(windowToTeamRwMutex_);
    auto it = windows_.find(handle);
    CHK_PRT_RET(it == windows_.end(),
        HCCL_WARNING("[WindowDeregister] window handle[%p] not found, maybe already destroyed", handle),
        HCOMM_SUCCESS);

    FreeWindowResources(it->second.get());
    windowToTeamMap_.erase(handle);
    windows_.erase(it);

    HCCL_INFO("[WindowDeregister] window[%p] destroyed, team[%p]", handle, team);
    return HCOMM_SUCCESS;
}

void HcommTeamMgr::MergeChannelLists(TeamEntry *entry, const HcommTeamBindChannelsDesc *desc,
                                        std::vector<std::vector<uint64_t>> &newChannels)
{
    if (entry->channelsList.empty()) {
        newChannels.resize(entry->hostTeam.memberNum);
    } else {
        newChannels = entry->channelsList;
    }

    for (uint32_t i = 0; i < entry->hostTeam.memberNum; i++) {
        uint64_t chNum = desc->channelNumPerMember[i];
        uint64_t *chSrc = desc->channelsByMemberId[i];
        for (uint64_t j = 0; j < chNum; j++) {
            newChannels[i].push_back(chSrc[j]);
        }
    }
}

HcommResult HcommTeamMgr::BindChannels(HcommTeamHandle team, const HcommTeamBindChannelsDesc *desc)
{
    std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
    TeamEntry *entry = FindTeamByHandleLocked(team);
    CHK_PRT_RET(entry == nullptr,
        HCCL_ERROR("[BindChannels] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);

    CHK_PRT_RET(desc->memberNum != entry->hostTeam.memberNum,
        HCCL_ERROR("[BindChannels] memberNum[%u] != team memberNum[%u]",
            desc->memberNum, entry->hostTeam.memberNum), HCOMM_E_PARA);

    for (uint32_t i = 0; i < entry->hostTeam.memberNum; i++) {
        CHK_PRT_RET(desc->channelNumPerMember[i] != 0 && desc->channelsByMemberId[i] == nullptr,
            HCCL_ERROR("[BindChannels] member[%u] channelNum[%u] > 0 but channels is null",
                i, desc->channelNumPerMember[i]), HCOMM_E_PARA);
    }

    // 二维数组，需要填入HcommTeam中的channels字段
    std::vector<std::vector<uint64_t>> newChannels;
    MergeChannelLists(entry, desc, newChannels);

    std::vector<std::vector<uint64_t>> oldChannels = std::move(entry->channelsList);
    entry->channelsList = std::move(newChannels);

    HcommResult ret = AllocAndCopyChannels(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[BindChannels] AllocAndCopyChannels failed, this bind does not take effect, "
                   "previously bound channels are not affected");
        entry->channelsList = std::move(oldChannels);
        return ret;
    }

    ret = SyncTeamToDevice(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[BindChannels] SyncTeamToDevice failed");
        return ret;
    }

    HCCL_INFO("[BindChannels] channels appended and synced, team[%p], channelNumPerMember[%p], "
        "channelsByMemberId[%p], memberNum[%u], isSubTeam[%d]",
        team, desc->channelNumPerMember, desc->channelsByMemberId,
        entry->hostTeam.memberNum, entry->isSubTeam);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::BindSyncMem(HcommTeamHandle team, const HcommTeamBindSyncMemDesc *desc)
{
    std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
    TeamEntry *entry = FindTeamByHandleLocked(team);
    CHK_PRT_RET(entry == nullptr,
        HCCL_ERROR("[BindSyncMem] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);

    CHK_PRT_RET(desc->remoteMemNum != entry->hostTeam.memberNum,
        HCCL_ERROR("[BindSyncMem] remoteMemNum[%u] != memberNum[%u]",
            desc->remoteMemNum, entry->hostTeam.memberNum),
        HCOMM_E_PARA);

    HcommResult ret = AllocAndCopyRemoteMems(entry, desc->remoteMems, desc->remoteMemNum);
    CHK_PRT_RET(ret != HCOMM_SUCCESS,
        HCCL_ERROR("[BindSyncMem] AllocAndCopyRemoteMems failed"), ret);

    if (entry->hostTeam.syncMem.shadowMem.addr != nullptr) {
        (void)hrtFree(entry->hostTeam.syncMem.shadowMem.addr);
        entry->hostTeam.syncMem.shadowMem.addr = nullptr;
        entry->hostTeam.syncMem.shadowMem.size = 0;
    }
    void *devShadowMemPtr = nullptr;
    uint64_t shadowMemSize = entry->hostTeam.syncMem.syncMemSize;
    ret = static_cast<HcommResult>(hrtMalloc(&devShadowMemPtr, shadowMemSize));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[BindSyncMem] hrtMalloc shadowMem failed, size[%llu] ret[%d]", shadowMemSize, ret);
        return ret;
    }
    entry->hostTeam.syncMem.shadowMem.type = COMM_MEM_TYPE_DEVICE;
    entry->hostTeam.syncMem.shadowMem.addr = devShadowMemPtr;
    entry->hostTeam.syncMem.shadowMem.size = shadowMemSize;

    ret = SyncTeamToDevice(entry);
    CHK_PRT_RET(ret != HCOMM_SUCCESS,
        HCCL_ERROR("[BindSyncMem] SyncTeamToDevice failed"), ret);

    HCCL_INFO("[BindSyncMem] syncMem bound, remoteMems[%p], remoteMemNum[%u]",
        desc->remoteMems, desc->remoteMemNum);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::GetNetLayer(HcommTeamHandle team, uint32_t *netLayer)
{
    std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
    TeamEntry *entry = FindTeamByHandleLocked(team);
    CHK_PRT_RET(entry == nullptr,
        HCCL_ERROR("[GetNetLayer] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);

    *netLayer = entry->hostTeam.netLayer;
    return HCOMM_SUCCESS;
}

} // namespace hcomm
