/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_TEAM_MGR_H
#define HCOMM_TEAM_MGR_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "hcomm_res.h"
#include "hcomm_res_defs.h"
#include "hcomm_team.h"
#include "hcomm_team_defs.h"
#include "hcomm_team_entity_defs.h"

namespace hcomm {
// HcommTeam / HcommWindow 结构体的 ABI 头部常量
constexpr uint32_t HCOMM_TEAM_MAGIC_WORD = 0x0f0f0f20U;
constexpr uint32_t HCOMM_TEAM_VERSION = 1U;
constexpr uint32_t HCOMM_WINDOW_MAGIC_WORD = 0x0f0f0f21U;
constexpr uint32_t HCOMM_WINDOW_VERSION = 1U;

struct WindowEntry {
    HcommWindow hostWindow{};
    HcommWindowHandle devWindow{nullptr};
    void* devMems{nullptr};
    CommMem* hostMems{nullptr};
    HcommTeamHandle teamHandle{nullptr};
};

struct TeamEntry {
    HcommTeam hostTeam{};
    HcommTeamHandle devTeam{nullptr};
    void* devWorldTeamIds{nullptr};
    void* devChannelNums{nullptr};
    void* devChannels{nullptr};
    void* devRemoteMems{nullptr};

    uint32_t* hostWorldTeamIds{nullptr};
    std::vector<uint32_t> hostChannelNums;
    std::vector<std::vector<uint64_t>> channelsList;

    CommMem* hostRemoteMems{nullptr};

    HcommTeamHandle worldTeamHandle{nullptr};
    bool isSubTeam{false};

    HcommTeamSyncMemRequirement syncMemReq{};
    uint64_t syncMemSize{0};
};

class HcommTeamMgr {
public:
    static HcommTeamMgr& GetInstance();

    HcommResult TeamCreate(
        HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize);
    HcommResult TeamDestroy(HcommTeamHandle team);
    HcommResult WindowRegister(HcommTeamHandle team, HcommWindowHandle* handle);
    HcommResult BindWindow(HcommTeamHandle team, HcommWindowHandle handle, const HcommTeamWindowDesc* desc);
    HcommResult WindowDeregister(HcommTeamHandle team, HcommWindowHandle handle);
    HcommResult BindChannels(HcommTeamHandle team, const HcommTeamBindChannelsDesc* desc);
    HcommResult BindSyncMem(HcommTeamHandle team, const HcommTeamBindSyncMemDesc* desc);
    HcommResult GetNetLayer(HcommTeamHandle team, uint32_t* netLayer);

private:
    HcommTeamMgr() = default;
    ~HcommTeamMgr();
    HcommTeamMgr(const HcommTeamMgr&) = delete;
    HcommTeamMgr& operator=(const HcommTeamMgr&) = delete;

    TeamEntry* FindTeamByHandleLocked(HcommTeamHandle handle);
    WindowEntry* FindWindowByHandleLocked(HcommWindowHandle handle);
    HcommResult SyncTeamToDevice(TeamEntry* entry);
    HcommResult SyncWindowToDevice(WindowEntry* entry);
    HcommResult AllocAndCopyWorldTeamIds(TeamEntry* entry, const uint32_t* src, uint32_t memberNum);
    HcommResult AllocAndCopyChannels(TeamEntry* entry);
    HcommResult AllocChannelEntities(TeamEntry* entry);
    HcommResult AllocChannelNumsArray(TeamEntry* entry);
    HcommResult AllocAndCopyRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum);
    // 首次分配 hostRemoteMems（calloc+memcpy）与 devRemoteMems（hrtMalloc+hrtMemSyncCopy），并写入 syncMem 字段
    HcommResult AllocRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum);
    // 重复 bind：校验维度一致后更新 hostRemoteMems（memcpy）并重新 sync 到 devRemoteMems
    HcommResult UpdateRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum);
    HcommResult ValidateSubTeam(TeamEntry* worldEntry, const HcommTeamCreateDesc* desc);
    void InitTeamEntry(TeamEntry* entry, const HcommTeamCreateDesc* desc, HcommTeamHandle worldTeam);
    HcommResult AllocAndSyncTeam(TeamEntry* entry, const HcommTeamCreateDesc* desc);
    HcommResult AllocAndCopyWindowMems(WindowEntry* winEntry, uint64_t memberNum, const CommMem* src);
    HcommResult MergeWindowMems(WindowEntry* winEntry, uint64_t memberNum, const CommMem* src);
    void MergeChannelLists(
        TeamEntry* entry, const HcommTeamBindChannelsDesc* desc, std::vector<std::vector<uint64_t>>& newChannels);
    void FreeTeamResources(TeamEntry* entry);
    void FreeWindowResources(WindowEntry* entry);
    void FreeDeviceChannels(TeamEntry* entry);

    std::shared_mutex teamsRwMutex_; // 保护 teams_
    std::unordered_map<HcommTeamHandle, std::unique_ptr<TeamEntry>> teams_;
    std::shared_mutex windowsRwMutex_; // 保护 windows_
    std::unordered_map<HcommWindowHandle, std::unique_ptr<WindowEntry>> windows_;
    std::shared_mutex windowToTeamRwMutex_; // 保护 windowToTeamMap_
    std::unordered_map<HcommWindowHandle, HcommTeamHandle> windowToTeamMap_;
};

} // namespace hcomm

#endif // HCOMM_TEAM_MGR_H
