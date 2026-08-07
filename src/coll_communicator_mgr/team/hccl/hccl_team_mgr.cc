/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_team_mgr.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "adapter_rts_common.h"
#include "log.h"

namespace hccl {

HcclTeamMgr& HcclTeamMgr::GetInstance()
{
    static std::once_flag instanceFlag;
    static HcclTeamMgr* instance = nullptr;
    std::call_once(instanceFlag, [&] {
        instance = new HcclTeamMgr();
    });
    return *instance;
}

HcclResult HcclTeamMgr::RegisterWorldTeam(
    HcommTeamHandle worldTeam, CollComm* collComm, void* syncMemPtr, uint64_t syncMemSize, const uint32_t* rankIds,
    uint32_t rankNum)
{
    if (worldTeam == nullptr || collComm == nullptr || rankIds == nullptr) {
        HCCL_ERROR(
            "[HcclTeamMgr][%s] invalid param, worldTeam[%p] collComm[%p] rankIds[%p]", __func__, worldTeam, collComm,
            rankIds);
        return HCCL_E_PTR;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto worldIt = teamMap_.find(worldTeam);
    if (worldIt != teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] worldTeam[%p] already registered", __func__, worldTeam);
        return HCCL_E_PARA;
    }
    TeamEntry& entry = teamMap_[worldTeam];
    entry.collComm = collComm;
    entry.worldTeam = nullptr;
    entry.syncMemPtr = syncMemPtr;
    entry.syncMemSize = syncMemSize;
    entry.syncMemHandle = nullptr;
    entry.syncMemTag.clear();
    entry.windows.clear();
    entry.rankIds.assign(rankIds, rankIds + rankNum);
    return HCCL_SUCCESS;
}

HcclResult HcclTeamMgr::RegisterSubTeam(
    HcommTeamHandle worldTeam, HcommTeamHandle subTeam, void* syncMemPtr, uint64_t syncMemSize, const uint32_t* rankIds,
    uint32_t rankNum)
{
    if (worldTeam == nullptr || subTeam == nullptr || rankIds == nullptr) {
        HCCL_ERROR(
            "[HcclTeamMgr][%s] invalid param, worldTeam[%p] subTeam[%p] rankIds[%p]", __func__, worldTeam, subTeam,
            rankIds);
        return HCCL_E_PTR;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto worldIt = teamMap_.find(worldTeam);
    if (worldIt == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] worldTeam[%p] not registered", __func__, worldTeam);
        return HCCL_E_PARA;
    }
    if (teamMap_.find(subTeam) != teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] subTeam[%p] already registered", __func__, subTeam);
        return HCCL_E_PARA;
    }
    CollComm* collComm = worldIt->second.collComm;
    TeamEntry& entry = teamMap_[subTeam];
    entry.collComm = collComm;
    entry.worldTeam = worldTeam;
    entry.syncMemPtr = syncMemPtr;
    entry.syncMemSize = syncMemSize;
    entry.syncMemHandle = nullptr;
    entry.syncMemTag.clear();
    entry.windows.clear();
    entry.rankIds.assign(rankIds, rankIds + rankNum);
    return HCCL_SUCCESS;
}

void HcclTeamMgr::UnregisterTeam(HcommTeamHandle team)
{
    if (team == nullptr) {
        return;
    }
    void* syncMemPtr = nullptr;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = teamMap_.find(team);
        if (it == teamMap_.end()) {
            return;
        }
        syncMemPtr = it->second.syncMemPtr;
        teamMap_.erase(it);
    }
    if (syncMemPtr != nullptr) {
        (void)hrtFree(syncMemPtr);
    }
}

CollComm* HcclTeamMgr::FindCollComm(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), nullptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return nullptr;
    }
    return it->second.collComm;
}

HcommTeamHandle HcclTeamMgr::FindWorldTeam(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), nullptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return nullptr;
    }
    // world team 自身 worldTeam 字段为 nullptr，返回自身
    return (it->second.worldTeam != nullptr) ? it->second.worldTeam : team;
}

std::vector<uint32_t> HcclTeamMgr::GetRankIds(HcommTeamHandle team)
{
    std::vector<uint32_t> result;
    if (team == nullptr) {
        return result;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        return result;
    }
    result = it->second.rankIds;
    return result;
}

void* HcclTeamMgr::GetSyncMemPtr(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), nullptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return nullptr;
    }
    return it->second.syncMemPtr;
}

uint64_t HcclTeamMgr::GetSyncMemSize(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), 0);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return 0;
    }
    return it->second.syncMemSize;
}

void HcclTeamMgr::SetTeamSyncMemHandle(HcommTeamHandle team, HcclMemHandle handle, const std::string& tag)
{
    if (team == nullptr) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team);
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return;
    }
    it->second.syncMemHandle = handle;
    it->second.syncMemTag = tag;
}

HcclMemHandle HcclTeamMgr::GetTeamSyncMemHandle(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), nullptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return nullptr;
    }
    return it->second.syncMemHandle;
}

std::string HcclTeamMgr::GetTeamSyncMemTag(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), std::string());
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return std::string();
    }
    return it->second.syncMemTag;
}

bool HcclTeamMgr::FindReusableWindow(HcommTeamHandle worldTeam, const CommMem& localMem, HcommWindowHandle& window)
{
    if (worldTeam == nullptr) {
        return false;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(worldTeam);
    if (it == teamMap_.end()) {
        return false;
    }
    // windows 按 registeredLocalMem.addr 升序
    // upper_bound 找第一个 start > requestStart 的，反向遍历找包含请求区间的 window（请求是已注册 window 的子集）。
    const uintptr_t requestStart = reinterpret_cast<uintptr_t>(localMem.addr);
    const uintptr_t requestEnd = requestStart + localMem.size;
    auto& windows = it->second.windows;
    auto upper
        = std::upper_bound(windows.begin(), windows.end(), requestStart, [](uintptr_t addr, const WindowInfo& win) {
              return addr < reinterpret_cast<uintptr_t>(win.registeredLocalMem.addr);
          });
    for (auto wIt = upper; wIt != windows.begin();) {
        --wIt;
        if (wIt->handle == nullptr) {
            continue;
        }
        const uintptr_t winStart = reinterpret_cast<uintptr_t>(wIt->registeredLocalMem.addr);
        const uintptr_t winEnd = winStart + wIt->registeredLocalMem.size;
        // 请求区间是已注册 window 的子集（含精确匹配）→ 复用
        if (requestStart >= winStart && requestEnd <= winEnd) {
            window = wIt->handle;
            // 精确匹配（addr+size 完全相同）→ 已有记录，无需新增
            if (requestStart == winStart && requestEnd == winEnd) {
                return true;
            }
            // 子集包含但非精确匹配 → 复用 parent handle，为请求的 localMem 新增一条 WindowInfo，
            // 使后续 BindWindowsAndSyncMem 的 self 槽填入请求的实际地址（而非 parent 的地址范围）。
            WindowInfo info;
            info.handle = window;
            info.registeredLocalMem = localMem;
            auto insertIt = std::upper_bound(
                windows.begin(), windows.end(), requestStart, [](uintptr_t addr, const WindowInfo& win) {
                    return addr < reinterpret_cast<uintptr_t>(win.registeredLocalMem.addr);
                });
            windows.insert(insertIt, std::move(info));
            return true;
        }
    }
    return false;
}

void HcclTeamMgr::AddWorldTeamWindow(
    HcommTeamHandle worldTeam, HcommWindowHandle window, const CommMem& localMem, HcclMemHandle localMemHandle,
    const std::string& localMemTag)
{
    if (worldTeam == nullptr || window == nullptr) {
        HCCL_ERROR("[HcclTeamMgr][%s] worldTeam[%p] or window[%p] is invalid", __func__, worldTeam, window);
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(worldTeam);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] worldTeam[%p] not registered", __func__, worldTeam);
        return;
    }
    WindowInfo info;
    info.handle = window;
    info.registeredLocalMem = localMem;
    info.localMemHandle = localMemHandle;
    info.localMemTag = localMemTag;
    // 按 registeredLocalMem.addr 升序插入，维护 windows 有序性，供 FindReusableWindow 二分查找
    const uintptr_t newStart = reinterpret_cast<uintptr_t>(localMem.addr);
    auto insertIt = std::upper_bound(
        it->second.windows.begin(), it->second.windows.end(), newStart, [](uintptr_t addr, const WindowInfo& win) {
            return addr < reinterpret_cast<uintptr_t>(win.registeredLocalMem.addr);
        });
    it->second.windows.insert(insertIt, std::move(info));
}

std::vector<WindowInfo> HcclTeamMgr::GetWorldTeamWindows(HcommTeamHandle worldTeam)
{
    std::vector<WindowInfo> result;
    if (worldTeam == nullptr) {
        return result;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(worldTeam);
    if (it == teamMap_.end()) {
        return result;
    }
    result = it->second.windows;
    return result;
}

std::vector<HcclMemHandle> HcclTeamMgr::CollectPendingMemHandles(HcommTeamHandle worldTeam, HcommTeamHandle team)
{
    std::vector<HcclMemHandle> result;
    if (worldTeam == nullptr || team == nullptr) {
        return result;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto itTeam = teamMap_.find(team);
    if (itTeam == teamMap_.end()) {
        return result;
    }
    auto itWorldTeam = teamMap_.find(worldTeam);
    if (itWorldTeam == teamMap_.end()) {
        return result;
    }
    // syncMemHandle：首次注册且未交换时收集
    if (!itTeam->second.syncMemExchanged && itTeam->second.syncMemHandle != nullptr) {
        result.push_back(itTeam->second.syncMemHandle);
        itTeam->second.syncMemExchanged = true;
    }
    // window localMemHandle：未交换的收集并标记
    for (auto& win : itWorldTeam->second.windows) {
        if (!win.exchanged && win.localMemHandle != nullptr) {
            result.push_back(win.localMemHandle);
            win.exchanged = true;
        }
    }
    return result;
}

void HcclTeamMgr::RemoveWorldTeamWindow(HcommTeamHandle worldTeam, HcommWindowHandle window)
{
    if (worldTeam == nullptr || window == nullptr) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(worldTeam);
    if (it == teamMap_.end()) {
        return;
    }
    auto& wins = it->second.windows;
    for (auto wIt = wins.begin(); wIt != wins.end(); ++wIt) {
        if (wIt->handle == window) {
            wins.erase(wIt);
            return;
        }
    }
}

std::vector<HcommTeamHandle> HcclTeamMgr::GetSubTeams(HcommTeamHandle worldTeam)
{
    std::vector<HcommTeamHandle> result;
    if (worldTeam == nullptr) {
        return result;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    // subTeam 的 worldTeam 字段指向其父 worldTeam；worldTeam 自身该字段为 nullptr。
    for (const auto& pair : teamMap_) {
        if (pair.second.worldTeam == worldTeam) {
            result.push_back(pair.first);
        }
    }
    return result;
}

void HcclTeamMgr::ClearByCollComm(CollComm* collComm)
{
    if (collComm == nullptr) {
        return;
    }
    // 锁内收集并 erase，锁外销毁，避免持 L2 锁调 L3 接口（锁嵌套/死锁风险）
    std::vector<TeamCleanupInfo> cleanupInfos = CollectTeamCleanupInfo(collComm);
    ExecuteTeamCleanup(cleanupInfos);
}

std::vector<HcclTeamMgr::TeamCleanupInfo> HcclTeamMgr::CollectTeamCleanupInfo(CollComm* collComm)
{
    std::vector<TeamCleanupInfo> cleanupInfos;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (auto it = teamMap_.begin(); it != teamMap_.end();) {
        if (it->second.collComm != collComm) {
            ++it;
            continue;
        }
        TeamCleanupInfo info;
        info.handle = it->first;
        info.syncMemPtr = it->second.syncMemPtr;
        // window 归 worldTeam 所有（entry.worldTeam==nullptr 表示自身是 worldTeam）
        if (it->second.worldTeam == nullptr) {
            for (const auto& win : it->second.windows) {
                if (win.handle != nullptr) {
                    info.windows.push_back(win.handle);
                }
            }
        }
        cleanupInfos.push_back(std::move(info));
        it = teamMap_.erase(it);
    }
    return cleanupInfos;
}

void HcclTeamMgr::ExecuteTeamCleanup(const std::vector<TeamCleanupInfo>& cleanupInfos)
{
    // 锁外依次销毁：window（L3 devWindow/devMems）→ team（L3 device 资源）→ syncMem（L2 本地内存）
    for (const auto& info : cleanupInfos) {
        for (HcommWindowHandle win : info.windows) {
            (void)HcommTeamWindowDeregister(info.handle, win);
        }
        (void)HcommTeamDestroy(info.handle);
        if (info.syncMemPtr != nullptr) {
            (void)hrtFree(info.syncMemPtr);
        }
    }
}

} // namespace hccl
