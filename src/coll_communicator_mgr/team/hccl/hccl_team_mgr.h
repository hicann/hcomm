/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TEAM_MGR_H
#define HCCL_TEAM_MGR_H

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "coll_comm.h"
#include "hccl/hccl_res.h"
#include "hcomm_team.h"
#include "hcomm_team_defs.h"

namespace hccl {
/**
 * @note 职责：进程级 team 管理器，统一维护：
 *       1) WorldTeam 与 SubTeam 的父子关系（world→sub 列表 / sub→world）；
 *       2) Team 粒度的 syncMem 本地内存（HcommTeamCreate 返回大小后 hrtMalloc，待 channel 交换）；
 *       3) WorldTeam 与 Window 的 1:N 关系（worldTeam 存 window 列表），Team 与 Window 的 N:N 关系
 *          （ChannelsCreate 通过 FindWorldTeam 解析 worldTeam 的所有 window）。
 */
// 单个 window 的信息（归 worldTeam 所有，1:N）
typedef struct {
    HcommWindowHandle handle{nullptr};        // 业务 window 句柄
    CommMem registeredLocalMem{};             // 注册时的 localMem，判重基准（子集复用）
    HcclMemHandle localMemHandle{nullptr};    // localMem 注册得到的句柄
    std::string localMemTag;                  // localMem 的 memTag（ChannelsCreate 远端 tag 匹配用）
    bool exchanged{false};                    // 是否已参与建链交换，避免重复交换
} WindowInfo;

typedef struct {
    CollComm *collComm{nullptr};   // 反查通信域
    HcommTeamHandle worldTeam{nullptr}; // 父 world team；world team 自身为 nullptr
    void *syncMemPtr{nullptr};   // 本地 syncMem 内存指针（hrtMalloc 申请）
    uint64_t syncMemSize{0};     // syncMem 内存大小
    // team 粒度 syncMem 内存句柄（首次 WindowRegister 时注册一次）
    HcclMemHandle syncMemHandle{nullptr};
    std::string syncMemTag;
    bool syncMemExchanged{false};            // syncMemHandle 是否已参与建链交换，避免重复交换
    // worldTeam 下注册的所有 window（1:N，仅 world team 条目填充）
    std::vector<WindowInfo> windows;
    // memberId→rankId 映射，下标=memberId，值=rankId。L2 维护，rankId 不下沉 L3。
    std::vector<uint32_t> rankIds;
} TeamEntry;

class HcclTeamMgr {
public:
    static HcclTeamMgr &GetInstance();

    // —— 创建/销毁 team 时调用 ——
    // 注册 world team：存 collComm + syncMem + rankIds（memberId→rankId 映射）。
    HcclResult RegisterWorldTeam(HcommTeamHandle worldTeam, CollComm *collComm, void *syncMemPtr, uint64_t syncMemSize,
                                 const uint32_t *rankIds, uint32_t rankNum);
    // 注册 sub team：校验 worldTeam 存在，建父子关系，存 syncMem + rankIds（collComm 取自 worldTeam 条目）。
    HcclResult RegisterSubTeam(HcommTeamHandle worldTeam, HcommTeamHandle subTeam, void *syncMemPtr, uint64_t syncMemSize,
                               const uint32_t *rankIds, uint32_t rankNum);
    // 销毁 team：hrtFree syncMem + erase 自身。memHandles 不注销。
    void UnregisterTeam(HcommTeamHandle team);

    // —— 查询 ——
    CollComm *FindCollComm(HcommTeamHandle team);
    // sub→world；world team 自身返回自身，未找到返回 nullptr。
    HcommTeamHandle FindWorldTeam(HcommTeamHandle team);
    // 取 team 的 rankIds（memberId→rankId 映射）拷贝，未找到返回空。
    std::vector<uint32_t> GetRankIds(HcommTeamHandle team);

    // —— syncMem ——
    void *GetSyncMemPtr(HcommTeamHandle team);
    uint64_t GetSyncMemSize(HcommTeamHandle team);

    // —— team 粒度 syncMem 内存句柄（ChannelsCreate 首次注册并取用）——
    void SetTeamSyncMemHandle(HcommTeamHandle team, HcclMemHandle handle, const std::string &tag);
    HcclMemHandle GetTeamSyncMemHandle(HcommTeamHandle team);
    std::string GetTeamSyncMemTag(HcommTeamHandle team);

    // —— window 复用与判重（worldTeam 范围，HcclTeamWindowRegister 用）——
    // 遍历 worldTeam 的所有 window，找 registeredLocalMem 是入参 localMem 超集的 window。
    // 命中返回 true 并填充 window；否则 false。
    bool FindReusableWindow(HcommTeamHandle worldTeam, const CommMem &localMem, HcommWindowHandle &window);
    // 往 worldTeam 的 window 列表追加一条记录（HcclTeamWindowRegister 新建 window 后调用）。
    void AddWorldTeamWindow(HcommTeamHandle worldTeam, HcommWindowHandle window, const CommMem &localMem,
                            HcclMemHandle localMemHandle, const std::string &localMemTag);
    // 取 worldTeam 的所有 window（拷贝），供 ChannelsCreate 遍历绑定。
    std::vector<WindowInfo> GetWorldTeamWindows(HcommTeamHandle worldTeam);
    // 收集未交换的 memHandles（syncMemHandle + window localMemHandle），标记已交换，避免重复建链交换。
    std::vector<HcclMemHandle> CollectPendingMemHandles(HcommTeamHandle worldTeam, HcommTeamHandle team);
    // 从 worldTeam 的 window 列表移除指定 window 的记录（HcclTeamWindowDeregister 用）。
    void RemoveWorldTeamWindow(HcommTeamHandle worldTeam, HcommWindowHandle window);

    // 取 worldTeam 下所有 subTeam 的 handle（worldTeam 销毁时连带销毁用）。锁内收集，调用方锁外销毁。
    std::vector<HcommTeamHandle> GetSubTeams(HcommTeamHandle worldTeam);

    // —— CollComm 析构兜底：清理属于该 comm 的所有 team 条目（hrtFree syncMem + erase）——
    void ClearByCollComm(CollComm *collComm);

private:
    HcclTeamMgr() = default;
    ~HcclTeamMgr() = default;
    HcclTeamMgr(const HcclTeamMgr &) = delete;
    HcclTeamMgr &operator=(const HcclTeamMgr &) = delete;

    // ClearByCollComm 的单 team 销毁信息（锁内收集，锁外销毁）
    struct TeamCleanupInfo {
        HcommTeamHandle handle{nullptr};
        std::vector<HcommWindowHandle> windows; // 仅 worldTeam 非空
        void *syncMemPtr{nullptr};
    };
    // 锁内收集该通信域下所有 team 的销毁信息并 erase teamMap 条目
    std::vector<TeamCleanupInfo> CollectTeamCleanupInfo(CollComm *collComm);
    // 锁外依次销毁：window（L3）→ team（L3）→ syncMem（L2 本地内存）
    void ExecuteTeamCleanup(const std::vector<TeamCleanupInfo> &cleanupInfos);

    std::unordered_map<HcommTeamHandle, TeamEntry> teamMap_;
    std::shared_mutex mutex_; // 读写锁：查询类接口多读，注册/销毁类接口独占写
};
}  // namespace hccl

#endif  // HCCL_TEAM_MGR_H
