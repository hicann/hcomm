/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <shared_mutex>

#define private public
#define protected public
#include "hcomm_team_entity_defs.h"
#include "hcomm_team_c_adpt.h"
#include "hcomm_team_mgr.h"
#undef protected
#undef private

#include "hcomm_team.h"
#include "hcomm_res_defs.h"
#include "hcomm_result_defs.h"
#include "adapter_rts_common.h"

using namespace hcomm;

static uint32_t g_hrtMallocFailAfter = UINT32_MAX;
static uint32_t g_hrtMallocCallCount = 0;
static uint32_t g_hrtMemSyncCopyFailAfter = UINT32_MAX;
static uint32_t g_hrtMemSyncCopyCallCount = 0;
static uint32_t g_hrtFreeCallCount = 0;

static void ResetStubCounters()
{
    g_hrtMallocFailAfter = UINT32_MAX;
    g_hrtMallocCallCount = 0;
    g_hrtMemSyncCopyFailAfter = UINT32_MAX;
    g_hrtMemSyncCopyCallCount = 0;
    g_hrtFreeCallCount = 0;
}

static HcclResult StubHrtMalloc(void **devPtr, u64 size, bool level2Address)
{
    (void)level2Address;
    g_hrtMallocCallCount++;
    if (g_hrtMallocCallCount > g_hrtMallocFailAfter) {
        return HCCL_E_INTERNAL;
    }
    *devPtr = malloc(static_cast<size_t>(size));
    return (*devPtr != nullptr) ? HCCL_SUCCESS : HCCL_E_PTR;
}

static HcclResult StubHrtFree(void *devPtr)
{
    g_hrtFreeCallCount++;
    free(devPtr);
    return HCCL_SUCCESS;
}

static HcclResult StubHrtMemSyncCopy(void *dst, uint64_t destMax, const void *src,
    uint64_t count, HcclRtMemcpyKind kind)
{
    (void)destMax;
    (void)kind;
    g_hrtMemSyncCopyCallCount++;
    if (g_hrtMemSyncCopyCallCount > g_hrtMemSyncCopyFailAfter) {
        return HCCL_E_INTERNAL;
    }
    if (dst == nullptr || src == nullptr) {
        return HCCL_E_PTR;
    }
    // AllocChannelEntities 的 D2D 拷贝 src 是 UT 填的伪 ChannelHandle（如 0x100），不可解引用；
    // 其余场景（SyncTeamToDevice/AllocChannelNumsArray 等）src 是真实 host 内存。
    // 伪地址直接跳过拷贝（UT 不验证 D2D 内容），避免 memcpy 触发 SEGFAULT。
    if (reinterpret_cast<uintptr_t>(src) < 0x10000 || reinterpret_cast<uintptr_t>(dst) < 0x10000) {
        return HCCL_SUCCESS;
    }
    memcpy(dst, src, static_cast<size_t>(count));
    return HCCL_SUCCESS;
}

class TestHcommTeam : public testing::Test {
public:
    void SetUp() override
    {
        ResetStubCounters();
        MOCKER(hrtMalloc).stubs().will(invoke(StubHrtMalloc));
        MOCKER(hrtFree).stubs().will(invoke(StubHrtFree));
        MOCKER(hrtMemSyncCopy).stubs().will(invoke(StubHrtMemSyncCopy));
    }

    void TearDown() override
    {
        CleanupSingleton();
        GlobalMockObject::verify();
    }

    void CleanupSingleton()
    {
        auto &mgr = HcommTeamMgr::GetInstance();
        {
            std::unique_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
            for (auto &pair : mgr.teams_) {
                if (pair.second != nullptr) {
                    mgr.FreeTeamResources(pair.second.get());
                }
            }
            mgr.teams_.clear();
        }
        {
            std::unique_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
            for (auto &pair : mgr.windows_) {
                if (pair.second != nullptr) {
                    mgr.FreeWindowResources(pair.second.get());
                }
            }
            mgr.windows_.clear();
        }
        {
            std::unique_lock<std::shared_mutex> lock(mgr.windowToTeamRwMutex_);
            mgr.windowToTeamMap_.clear();
        }
    }

    HcommResult CreateWorldTeam(uint32_t memberNum, uint32_t selfMemberId,
        HcommTeamHandle &team, uint64_t &syncMemSize)
    {
        return CreateWorldTeam(memberNum, selfMemberId, 1, team, syncMemSize);
    }

    HcommResult CreateWorldTeam(uint32_t memberNum, uint32_t selfMemberId,
        uint32_t netLayer, HcommTeamHandle &team, uint64_t &syncMemSize)
    {
        std::vector<uint32_t> ids(memberNum);
        for (uint32_t i = 0; i < memberNum; i++) {
            ids[i] = i;
        }
        HcommTeamCreateDesc desc{};
        desc.memberNum = memberNum;
        desc.selfMemberId = selfMemberId;
        desc.worldMemberIds = ids.data();
        desc.netLayer = netLayer;
        desc.requirement = {0, 0, 1, {}};
        return HcommTeamCreate(nullptr, &desc, &team, &syncMemSize);
    }

    void BindValidChannels(HcommTeamHandle team, uint32_t memberNum)
    {
        std::vector<uint32_t> chNums(memberNum, 1);
        std::vector<std::vector<uint64_t>> channels(memberNum, std::vector<uint64_t>{0x100});
        std::vector<uint64_t *> chPtrs;
        for (auto &ch : channels) {
            chPtrs.push_back(ch.data());
        }
        HcommTeamBindChannelsDesc desc{};
        desc.memberNum = memberNum;
        desc.channelNumPerMember = chNums.data();
        desc.channelsByMemberId = chPtrs.data();
        HcommTeamBindChannels(team, &desc);
    }

    void BindValidSyncMem(HcommTeamHandle team, uint32_t memberNum)
    {
        std::vector<CommMem> mems(memberNum);
        for (uint32_t i = 0; i < memberNum; i++) {
            mems[i].addr = malloc(1024);
            mems[i].size = 1024;
            mems[i].type = COMM_MEM_TYPE_DEVICE;
        }
        HcommTeamBindSyncMemDesc desc{};
        desc.remoteMems = mems.data();
        desc.remoteMemNum = memberNum;
        HcommTeamBindRemoteSyncMem(team, &desc);
        for (uint32_t i = 0; i < memberNum; i++) {
            free(mems[i].addr);
        }
    }

    void BindValidWindow(HcommTeamHandle team, HcommWindowHandle handle, uint32_t memberNum)
    {
        std::vector<CommMem> mems(memberNum);
        for (uint32_t i = 0; i < memberNum; i++) {
            mems[i].addr = malloc(1024);
            mems[i].size = 1024;
            mems[i].type = COMM_MEM_TYPE_DEVICE;
        }
        HcommTeamWindowDesc desc{};
        desc.mems = mems.data();
        desc.memberNum = memberNum;
        HcommTeamWindowBindRemoteMems(team, handle, &desc);
        for (uint32_t i = 0; i < memberNum; i++) {
            free(mems[i].addr);
        }
    }
};

// ==================== ut_team_001 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_8MembersValid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    HcommResult ret = CreateWorldTeam(8, 3, team, syncMemSize);
    EXPECT_EQ(ret, HCOMM_SUCCESS);
    EXPECT_NE(team, nullptr);
    EXPECT_EQ(syncMemSize, static_cast<uint64_t>((0 + 0 + 1) * sizeof(uint64_t) * 8));
}

// ==================== ut_team_002 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_MemberNum1_Expect_ReturnParaError)
{
    std::vector<uint32_t> ids = {0};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 1;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.netLayer = 0;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_E_PARA);
    EXPECT_EQ(team, nullptr);
}

// ==================== ut_team_003 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_DescNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, nullptr, &team, &syncMemSize), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_TeamOutNull_Expect_ReturnPtrError)
{
    HcommTeamCreateDesc desc{};
    desc.memberNum = 2;
    desc.selfMemberId = 0;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, nullptr, &syncMemSize), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SyncMemSizeOutNull_Expect_ReturnPtrError)
{
    HcommTeamCreateDesc desc{};
    desc.memberNum = 2;
    desc.selfMemberId = 0;
    HcommTeamHandle team = nullptr;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, &team, nullptr), HCOMM_E_PTR);
}

// ==================== ut_team_004 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_MemberNumZero_Expect_ReturnParaError)
{
    HcommTeamCreateDesc desc{};
    desc.memberNum = 0;
    desc.selfMemberId = 0;
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_E_PARA);
}

// ==================== ut_team_005 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SelfMemberIdExceeds_Expect_ReturnParaError)
{
    std::vector<uint32_t> ids = {0, 1, 2, 3};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 4;
    desc.worldMemberIds = ids.data();
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_E_PARA);
}

// ==================== ut_team_006 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamWorldMemberIdsNull_Expect_ReturnPtrError)
{
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 0;
    desc.worldMemberIds = nullptr;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    HcommTeamHandle fakeWorldTeam = reinterpret_cast<HcommTeamHandle>(0x10000);
    EXPECT_EQ(HcommTeamCreate(fakeWorldTeam, &desc, &team, &syncMemSize), HCOMM_E_PTR);
}

// ==================== ut_team_007 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamValid_Expect_ReturnSuccess)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(8, 3, worldTeam, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> subIds = {2, 5, 7};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 3;
    desc.selfMemberId = 1;
    desc.worldMemberIds = subIds.data();
    desc.netLayer = 0;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle subTeam = nullptr;
    uint64_t subWsSize = 0;
    EXPECT_EQ(HcommTeamCreate(worldTeam, &desc, &subTeam, &subWsSize), HCOMM_SUCCESS);
    EXPECT_NE(subTeam, nullptr);
}

// ==================== ut_team_008 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamInvalidWorldTeam_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fakeWorld = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    std::vector<uint32_t> ids = {0, 1, 2};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 3;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(fakeWorld, &desc, &team, &syncMemSize), HCOMM_E_NOT_FOUND);
}

// ==================== ut_team_009 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamMemberNumExceeds_Expect_ReturnParaError)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, worldTeam, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> ids(8, 0);
    HcommTeamCreateDesc desc{};
    desc.memberNum = 8;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    HcommTeamHandle team = nullptr;
    uint64_t wsSize2 = 0;
    EXPECT_EQ(HcommTeamCreate(worldTeam, &desc, &team, &wsSize2), HCOMM_E_PARA);
}

// ==================== ut_team_010 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamWorldMemberIdsOutOfBounds_Expect_ReturnParaError)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, worldTeam, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> ids = {1, 3, 5};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 3;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    HcommTeamHandle team = nullptr;
    uint64_t wsSize2 = 0;
    EXPECT_EQ(HcommTeamCreate(worldTeam, &desc, &team, &wsSize2), HCOMM_E_PARA);
}

// ==================== ut_team_011 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.find(team), mgr.teams_.end());
}

// ==================== ut_team_012 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_Null_Expect_ReturnPtrError)
{
    EXPECT_EQ(HcommTeamDestroy(nullptr), HCOMM_E_PTR);
}

// 重复销毁（handle 不存在）幂等返回 SUCCESS，仅打 WARNING
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_InvalidHandle_Expect_ReturnSuccess)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    EXPECT_EQ(HcommTeamDestroy(fake), HCOMM_SUCCESS);
}

// ==================== ut_team_013 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowRegister_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    EXPECT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);
    EXPECT_NE(handle, nullptr);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> winLock(mgr.windowsRwMutex_); std::shared_lock<std::shared_mutex> mapLock(mgr.windowToTeamRwMutex_);
    EXPECT_NE(mgr.windows_.find(handle), mgr.windows_.end());
    EXPECT_NE(mgr.windowToTeamMap_.find(handle), mgr.windowToTeamMap_.end());
}

// ==================== ut_team_014 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowRegister_When_TeamNull_Expect_ReturnPtrError)
{
    HcommWindowHandle handle = nullptr;
    EXPECT_EQ(HcommTeamWindowRegister(nullptr, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowRegister_When_HandleOutNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamWindowRegister(team, nullptr, nullptr, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_E_PTR);
}

// ==================== ut_team_015 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamWindowDesc desc{};
    desc.mems = mems.data();
    desc.memberNum = 4;
    EXPECT_EQ(HcommTeamWindowBindRemoteMems(team, handle, &desc), HCOMM_SUCCESS);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

// ==================== ut_team_016 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_TeamNull_Expect_ReturnPtrError)
{
    HcommWindowHandle handle = reinterpret_cast<HcommWindowHandle>(0x2000);
    HcommTeamWindowDesc desc{};
    EXPECT_EQ(HcommTeamWindowBindRemoteMems(nullptr, handle, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_HandleNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    HcommTeamWindowDesc desc{};
    EXPECT_EQ(HcommTeamWindowBindRemoteMems(team, nullptr, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_DescNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    HcommWindowHandle handle = reinterpret_cast<HcommWindowHandle>(0x2000);
    EXPECT_EQ(HcommTeamWindowBindRemoteMems(team, handle, nullptr), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_MemsNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    HcommWindowHandle handle = reinterpret_cast<HcommWindowHandle>(0x2000);
    HcommTeamWindowDesc desc{};
    desc.mems = nullptr;
    EXPECT_EQ(HcommTeamWindowBindRemoteMems(team, handle, &desc), HCOMM_E_PTR);
}

// ==================== ut_team_018 ====================
// 源码 MergeWindowMems：同 member 重复 bind（addr 非空覆盖已绑定槽）返回 HCOMM_E_PARA，需注册新 window 重新 bind。
TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_DuplicateBind_Expect_ReturnParaError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamWindowDesc desc{};
    desc.mems = mems.data();
    desc.memberNum = 4;
    ASSERT_EQ(HcommTeamWindowBindRemoteMems(team, handle, &desc), HCOMM_SUCCESS);
    // 重复 bind 同 member（addr 非空）：源码返回 HCOMM_E_PARA，需注册新 window 重新 bind。
    EXPECT_EQ(HcommTeamWindowBindRemoteMems(team, handle, &desc), HCOMM_E_PARA);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

// ==================== ut_team_019 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowDeregister_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamWindowDeregister(team, handle), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> winLock(mgr.windowsRwMutex_); std::shared_lock<std::shared_mutex> mapLock(mgr.windowToTeamRwMutex_);
    EXPECT_EQ(mgr.windows_.find(handle), mgr.windows_.end());
    EXPECT_EQ(mgr.windowToTeamMap_.find(handle), mgr.windowToTeamMap_.end());
}

// ==================== ut_team_020 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowDeregister_When_Null_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamWindowDeregister(team, nullptr), HCOMM_E_PTR);
}

// 无效 team handle：实现返回 NOT_FOUND（team 不存在）
TEST_F(TestHcommTeam, Ut_HcommTeamWindowDeregister_When_InvalidHandle_Expect_ReturnNotFoundError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    HcommWindowHandle fake = reinterpret_cast<HcommWindowHandle>(0xDEADBEEF);
    EXPECT_EQ(HcommTeamWindowDeregister(team, fake), HCOMM_E_NOT_FOUND);
}

// ==================== ut_team_021 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_FirstBindValid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> chNums = {2, 2, 2, 2};
    std::vector<std::vector<uint64_t>> channels = {
        {0x100, 0x200}, {0x300, 0x400}, {0x500, 0x600}, {0x700, 0x800}
    };
    std::vector<uint64_t *> chPtrs;
    for (auto &ch : channels) {
        chPtrs.push_back(ch.data());
    }
    HcommTeamBindChannelsDesc desc{};
    desc.memberNum = 4;
    desc.channelNumPerMember = chNums.data();
    desc.channelsByMemberId = chPtrs.data();
    EXPECT_EQ(HcommTeamBindChannels(team, &desc), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto *entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->channelsList.size(), 4u);
    EXPECT_EQ(entry->channelsList[0].size(), 2u);
    // devChannels 是 device 侧连续 channel 数组基地址；channelNumPerMember 是 device 侧每成员 channel 数数组。
    EXPECT_NE(entry->devChannels, nullptr);
    EXPECT_NE(entry->hostTeam.channelNumPerMember, nullptr);
}

// ==================== ut_team_022 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_SecondAppend_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> chNums1 = {1, 1, 1, 1};
    std::vector<std::vector<uint64_t>> channels1 = {{0x100}, {0x200}, {0x300}, {0x400}};
    std::vector<uint64_t *> chPtrs1;
    for (auto &ch : channels1) {
        chPtrs1.push_back(ch.data());
    }
    HcommTeamBindChannelsDesc desc1{};
    desc1.memberNum = 4;
    desc1.channelNumPerMember = chNums1.data();
    desc1.channelsByMemberId = chPtrs1.data();
    ASSERT_EQ(HcommTeamBindChannels(team, &desc1), HCOMM_SUCCESS);

    std::vector<uint32_t> chNums2 = {1, 1, 1, 1};
    std::vector<std::vector<uint64_t>> channels2 = {{0x500}, {0x600}, {0x700}, {0x800}};
    std::vector<uint64_t *> chPtrs2;
    for (auto &ch : channels2) {
        chPtrs2.push_back(ch.data());
    }
    HcommTeamBindChannelsDesc desc2{};
    desc2.memberNum = 4;
    desc2.channelNumPerMember = chNums2.data();
    desc2.channelsByMemberId = chPtrs2.data();
    EXPECT_EQ(HcommTeamBindChannels(team, &desc2), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto *entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->channelsList[0].size(), 2u);
    EXPECT_EQ(entry->channelsList[0][0], 0x100u);
    EXPECT_EQ(entry->channelsList[0][1], 0x500u);
}

// ==================== ut_team_023 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_TeamNull_Expect_ReturnPtrError)
{
    HcommTeamBindChannelsDesc desc{};
    EXPECT_EQ(HcommTeamBindChannels(nullptr, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_DescNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamBindChannels(team, nullptr), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_ChannelNumPerMemberNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    HcommTeamBindChannelsDesc desc{};
    EXPECT_EQ(HcommTeamBindChannels(team, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_ChannelsByMemberIdNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    HcommTeamBindChannelsDesc desc{};
    std::vector<uint32_t> chNums = {1, 1};
    desc.channelNumPerMember = chNums.data();
    EXPECT_EQ(HcommTeamBindChannels(team, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_MemberNumMismatch_Expect_ReturnParaError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommTeamBindChannelsDesc desc{};
    desc.memberNum = 8;
    std::vector<uint32_t> chNums(8, 1);
    std::vector<std::vector<uint64_t>> channels(8, std::vector<uint64_t>{0x100});
    std::vector<uint64_t *> chPtrs;
    for (auto &ch : channels) {
        chPtrs.push_back(ch.data());
    }
    desc.channelNumPerMember = chNums.data();
    desc.channelsByMemberId = chPtrs.data();
    EXPECT_EQ(HcommTeamBindChannels(team, &desc), HCOMM_E_PARA);
}

// ==================== ut_team_025 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 4;
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto *entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_NE(entry->devRemoteMems, nullptr);
    EXPECT_NE(entry->hostRemoteMems, nullptr);
    EXPECT_EQ(entry->hostTeam.syncMem.remoteMemsNum, 4u);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

// ==================== ut_team_026 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_ReplaceOld_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<CommMem> mems1(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems1[i].addr = malloc(1024);
        mems1[i].size = 1024;
        mems1[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamBindSyncMemDesc desc1{};
    desc1.remoteMems = mems1.data();
    desc1.remoteMemNum = 4;
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &desc1), HCOMM_SUCCESS);

    std::vector<CommMem> mems2(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems2[i].addr = malloc(2048);
        mems2[i].size = 2048;
        mems2[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamBindSyncMemDesc desc2{};
    desc2.remoteMems = mems2.data();
    desc2.remoteMemNum = 4;
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, &desc2), HCOMM_SUCCESS);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems1[i].addr);
        free(mems2[i].addr);
    }
}

// ==================== ut_team_027 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_TeamNull_Expect_ReturnPtrError)
{
    HcommTeamBindSyncMemDesc desc{};
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(nullptr, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_DescNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, nullptr), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_RemoteMemsNull_Expect_ReturnPtrError)
{
    // 源码：remoteMems 数组下标为 memberId、长度=memberNum（恒>0），故 remoteMems 必须非空，为 nullptr 返回 HCOMM_E_PTR。
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = nullptr;
    desc.remoteMemNum = 4;
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_RemoteMemNumMismatch_Expect_ReturnParaError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<CommMem> mems(8);
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 8;
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_E_PARA);
}

// ==================== ut_team_030 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamGetNetLayer_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, 1u, team, syncMemSize), HCOMM_SUCCESS);

    uint32_t netLayer = 0;
    EXPECT_EQ(HcommTeamGetNetLayer(team, &netLayer), HCOMM_SUCCESS);
    EXPECT_EQ(netLayer, 1u);
}

// ==================== ut_team_031 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamGetNetLayer_When_Null_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamGetNetLayer(team, nullptr), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamGetNetLayer_When_TeamNull_Expect_ReturnPtrError)
{
    uint32_t netLayer = 0;
    EXPECT_EQ(HcommTeamGetNetLayer(nullptr, &netLayer), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamGetNetLayer_When_InvalidHandle_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    uint32_t netLayer = 0;
    EXPECT_EQ(HcommTeamGetNetLayer(fake, &netLayer), HCOMM_E_NOT_FOUND);
}

// ==================== ut_team_032 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_HrtMallocFailAfterWorldTeamIds_Expect_NoLeak)
{
    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 1;

    std::vector<uint32_t> ids = {0, 1, 2, 3};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.netLayer = 1;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    HcommResult ret = HcommTeamCreate(nullptr, &desc, &team, &syncMemSize);
    EXPECT_NE(ret, HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.size(), 0u);
}

// ==================== ut_team_033 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_AppendFail_Expect_Rollback)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> chNums1 = {1, 1, 1, 1};
    std::vector<std::vector<uint64_t>> channels1 = {{0x100}, {0x200}, {0x300}, {0x400}};
    std::vector<uint64_t *> chPtrs1;
    for (auto &ch : channels1) {
        chPtrs1.push_back(ch.data());
    }
    HcommTeamBindChannelsDesc desc1{};
    desc1.memberNum = 4;
    desc1.channelNumPerMember = chNums1.data();
    desc1.channelsByMemberId = chPtrs1.data();
    ASSERT_EQ(HcommTeamBindChannels(team, &desc1), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::vector<std::vector<uint64_t>> oldChannels;
    {
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto *entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        oldChannels = entry->channelsList;
    }

    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;

    std::vector<uint32_t> chNums2 = {1, 1, 1, 1};
    std::vector<std::vector<uint64_t>> channels2 = {{0x500}, {0x600}, {0x700}, {0x800}};
    std::vector<uint64_t *> chPtrs2;
    for (auto &ch : channels2) {
        chPtrs2.push_back(ch.data());
    }
    HcommTeamBindChannelsDesc desc2{};
    desc2.memberNum = 4;
    desc2.channelNumPerMember = chNums2.data();
    desc2.channelsByMemberId = chPtrs2.data();
    HcommResult ret = HcommTeamBindChannels(team, &desc2);
    EXPECT_NE(ret, HCOMM_SUCCESS);

    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto *entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->channelsList.size(), oldChannels.size());
    for (size_t i = 0; i < oldChannels.size(); i++) {
        EXPECT_EQ(entry->channelsList[i], oldChannels[i]);
    }
}

// ==================== ut_team_034 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_BoundResources_Expect_AllFreed)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    BindValidChannels(team, 4);
    BindValidSyncMem(team, 4);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);
    BindValidWindow(team, handle, 4);

    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.find(team), mgr.teams_.end());
}

// ==================== ut_team_035 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowDeregister_When_BeforeTeamDestroy_Expect_BothSucceed)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    EXPECT_EQ(HcommTeamWindowDeregister(team, handle), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);
}

// ==================== ut_team_036 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_WindowExists_Expect_WindowDeregisterError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamWindowDeregister(team, handle), HCOMM_E_NOT_FOUND);
}

// ==================== ut_team_037 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_MultipleTeamsWindowsMixed_Expect_AllSucceed)
{
    HcommTeamHandle team1 = nullptr, team2 = nullptr;
    uint64_t wsSize1 = 0, wsSize2 = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team1, wsSize1), HCOMM_SUCCESS);
    ASSERT_EQ(CreateWorldTeam(8, 0, team2, wsSize2), HCOMM_SUCCESS);

    HcommWindowHandle win1 = nullptr, win2 = nullptr, win3 = nullptr, win4 = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team1, nullptr, &win1, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);
    ASSERT_EQ(HcommTeamWindowRegister(team1, nullptr, &win2, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);
    ASSERT_EQ(HcommTeamWindowRegister(team2, nullptr, &win3, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);
    ASSERT_EQ(HcommTeamWindowRegister(team2, nullptr, &win4, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    EXPECT_EQ(HcommTeamWindowDeregister(team1, win1), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(team1), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamWindowDeregister(team2, win3), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(team2), HCOMM_SUCCESS);
}

// ==================== ut_team_038 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_StructLayout_Expect_FieldOrderCorrect)
{
    EXPECT_LT(offsetof(HcommTeam, header), offsetof(HcommTeam, engine));
    EXPECT_LT(offsetof(HcommTeam, engine), offsetof(HcommTeam, memberNum));
    EXPECT_LT(offsetof(HcommTeam, memberNum), offsetof(HcommTeam, selfMemberId));
    EXPECT_LT(offsetof(HcommTeam, selfMemberId), offsetof(HcommTeam, channelsBaseAddr));
    EXPECT_LT(offsetof(HcommTeam, channelsBaseAddr), offsetof(HcommTeam, channelNumPerMember));
    EXPECT_LT(offsetof(HcommTeam, channelNumPerMember), offsetof(HcommTeam, netLayer));
    EXPECT_LT(offsetof(HcommTeam, netLayer), offsetof(HcommTeam, worldTeamIds));
    EXPECT_LT(offsetof(HcommTeam, worldTeamIds), offsetof(HcommTeam, syncMem));
    EXPECT_LT(offsetof(HcommTeam, syncMem), offsetof(HcommTeam, reserved));
}

// ==================== ut_team_039 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_HcommWindowStructLayout_Expect_FieldOrderCorrect)
{
    EXPECT_LT(offsetof(HcommWindow, header), offsetof(HcommWindow, memsNum));
    EXPECT_LT(offsetof(HcommWindow, memsNum), offsetof(HcommWindow, mems));
    EXPECT_LT(offsetof(HcommWindow, mems), offsetof(HcommWindow, reserved));
}

// ==================== ut_team_040 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_ConcurrentCreate_Expect_NoDataRace)
{
    const int threadNum = 4;
    std::vector<HcommTeamHandle> teams(threadNum, nullptr);
    std::vector<uint64_t> wsSizes(threadNum, 0);
    std::vector<HcommResult> results(threadNum, HCOMM_E_INTERNAL);

    std::vector<std::thread> threads;
    for (int i = 0; i < threadNum; i++) {
        threads.emplace_back([this, i, &teams, &wsSizes, &results]() {
            results[i] = CreateWorldTeam(4, 0, teams[i], wsSizes[i]);
        });
    }
    for (auto &t : threads) {
        t.join();
    }

    for (int i = 0; i < threadNum; i++) {
        EXPECT_EQ(results[i], HCOMM_SUCCESS);
        EXPECT_NE(teams[i], nullptr);
    }
}

// ==================== ut_team_041 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_ConcurrentCreateDestroy_Expect_NoCrash)
{
    const int threadNum = 4;
    std::vector<HcommResult> results(threadNum, HCOMM_E_INTERNAL);

    std::vector<std::thread> threads;
    for (int i = 0; i < threadNum; i++) {
        threads.emplace_back([this, i, &results]() {
            HcommTeamHandle team = nullptr;
            uint64_t syncMemSize = 0;
            HcommResult ret = CreateWorldTeam(4, 0, team, syncMemSize);
            if (ret == HCOMM_SUCCESS && team != nullptr) {
                ret = HcommTeamDestroy(team);
            }
            results[i] = ret;
        });
    }
    for (auto &t : threads) {
        t.join();
    }

    for (int i = 0; i < threadNum; i++) {
        EXPECT_EQ(results[i], HCOMM_SUCCESS);
    }
}

// ==================== ut_team_042 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_AfterBind_Expect_DeviceMemsPtrValid)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamWindowDesc desc{};
    desc.mems = mems.data();
    desc.memberNum = 4;
    ASSERT_EQ(HcommTeamWindowBindRemoteMems(team, handle, &desc), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    auto *winEntry = mgr.FindWindowByHandleLocked(handle);
    ASSERT_NE(winEntry, nullptr);
    EXPECT_NE(winEntry->hostWindow.mems, nullptr);
    EXPECT_EQ(winEntry->hostWindow.memsNum, 4u);
    EXPECT_NE(winEntry->devMems, nullptr);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

// ==================== ut_team_043 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_AfterBind_Expect_DeviceSyncMemValid)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 4;
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto *entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_NE(entry->hostTeam.syncMem.remoteMems, nullptr);
    EXPECT_EQ(entry->hostTeam.syncMem.remoteMemsNum, 4u);
    EXPECT_NE(entry->devRemoteMems, nullptr);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

// ==================== ut_team_044 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_SubTeamOnly_Expect_WorldTeamAccessible)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(8, 3, worldTeam, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> subIds = {2, 5, 7};
    HcommTeamCreateDesc subDesc{};
    subDesc.memberNum = 3;
    subDesc.selfMemberId = 1;
    subDesc.worldMemberIds = subIds.data();
    subDesc.netLayer = 0;
    subDesc.requirement = {0, 0, 1, {}};
    HcommTeamHandle subTeam = nullptr;
    uint64_t subWsSize = 0;
    ASSERT_EQ(HcommTeamCreate(worldTeam, &subDesc, &subTeam, &subWsSize), HCOMM_SUCCESS);

    EXPECT_EQ(HcommTeamDestroy(subTeam), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(worldTeam), HCOMM_SUCCESS);
}

// ==================== ut_team_045 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamWindowRegister_When_AfterRegister_Expect_DeviceWindowEmpty)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    auto *winEntry = mgr.FindWindowByHandleLocked(handle);
    ASSERT_NE(winEntry, nullptr);
    EXPECT_NE(winEntry->devWindow, nullptr);

    HcommWindow devWindow;
    memcpy(&devWindow, winEntry->devWindow, sizeof(HcommWindow));
    EXPECT_EQ(devWindow.memsNum, 0u);
    EXPECT_EQ(devWindow.mems, nullptr);
}

// ==================== Additional coverage tests ====================

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_HrtMallocFailForDevWorldTeamIds_Expect_ReturnError)
{
    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;

    std::vector<uint32_t> ids = {0, 1, 2, 3};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.netLayer = 1;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_NE(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.size(), 0u);
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_HrtMemSyncCopyFailInWorldTeamIds_Expect_ReturnError)
{
    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 0;

    std::vector<uint32_t> ids = {0, 1, 2, 3};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.netLayer = 1;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_NE(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.size(), 0u);
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SyncTeamToDeviceFail_Expect_ReturnError)
{
    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 1;

    std::vector<uint32_t> ids = {0, 1, 2, 3};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.netLayer = 1;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_NE(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.size(), 0u);
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowRegister_When_HrtMallocFail_Expect_ReturnError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;

    HcommWindowHandle handle = nullptr;
    EXPECT_NE(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowRegister_When_SyncWindowToDeviceFail_Expect_ReturnError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 0;

    HcommWindowHandle handle = nullptr;
    EXPECT_NE(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_HrtMallocDevMemsFail_Expect_ReturnError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamWindowDesc desc{};
    desc.mems = mems.data();
    desc.memberNum = 4;
    EXPECT_NE(HcommTeamWindowBindRemoteMems(team, handle, &desc), HCOMM_SUCCESS);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_SyncWindowToDeviceFail_Expect_ReturnError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamWindowDesc desc{};
    desc.mems = mems.data();
    desc.memberNum = 4;

    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 1;
    EXPECT_NE(HcommTeamWindowBindRemoteMems(team, handle, &desc), HCOMM_SUCCESS);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_HrtMallocFail_Expect_ReturnError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;

    std::vector<CommMem> mems(4);
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 4;
    EXPECT_NE(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_SUCCESS);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_SyncTeamToDeviceFail_Expect_ReturnError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 4;

    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 1;
    EXPECT_NE(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_SUCCESS);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_InvalidTeamHandle_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    HcommTeamBindChannelsDesc desc{};
    std::vector<uint32_t> chNums = {1, 1};
    desc.memberNum = 2;
    desc.channelNumPerMember = chNums.data();
    std::vector<uint64_t> ch0 = {0x100};
    std::vector<uint64_t> ch1 = {0x200};
    std::vector<uint64_t *> chPtrs = {ch0.data(), ch1.data()};
    desc.channelsByMemberId = chPtrs.data();
    EXPECT_EQ(HcommTeamBindChannels(fake, &desc), HCOMM_E_NOT_FOUND);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_InvalidTeamHandle_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    std::vector<CommMem> mems(2);
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 2;
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(fake, &desc), HCOMM_E_NOT_FOUND);
}

TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_InvalidHandles_Expect_ReturnNotFoundError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle fakeWin = reinterpret_cast<HcommWindowHandle>(0xDEADBEEF);
    std::vector<CommMem> mems(4);
    HcommTeamWindowDesc desc{};
    desc.mems = mems.data();
    desc.memberNum = 4;
    EXPECT_EQ(HcommTeamWindowBindRemoteMems(team, fakeWin, &desc), HCOMM_E_NOT_FOUND);
}

TEST_F(TestHcommTeam, Ut_HcommTeam_When_HcommTeamSyncMemStructLayout_Expect_FieldOrderCorrect)
{
    EXPECT_LT(offsetof(HcommTeamSyncMem, remoteMems), offsetof(HcommTeamSyncMem, remoteMemsNum));
    EXPECT_LT(offsetof(HcommTeamSyncMem, remoteMemsNum), offsetof(HcommTeamSyncMem, syncMemReq));
    EXPECT_LT(offsetof(HcommTeamSyncMem, syncMemReq), offsetof(HcommTeamSyncMem, syncMemSize));
    EXPECT_LT(offsetof(HcommTeamSyncMem, syncMemSize), offsetof(HcommTeamSyncMem, reserved));
}

// ==================== ut_team_046 ====================
// 验证重复 BindSyncMem 时旧 shadowMem 被释放，不会因覆盖而泄漏
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_Rebind_Expect_OldShadowMemFreed)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 4;

    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_SUCCESS);
    auto &mgr = HcommTeamMgr::GetInstance();
    void *firstShadow = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto *entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        firstShadow = entry->hostTeam.syncMem.shadowMem.addr;
        ASSERT_NE(firstShadow, nullptr);
    }
    uint32_t freeBeforeRebind = g_hrtFreeCallCount;

    // 重复 bind：旧 shadowMem 必须先被释放，再申请新的
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_SUCCESS);
    void *secondShadow = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto *entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        secondShadow = entry->hostTeam.syncMem.shadowMem.addr;
        ASSERT_NE(secondShadow, nullptr);
    }
    // 重复 bind 至少触发一次额外 free（释放旧 shadowMem）
    EXPECT_GT(g_hrtFreeCallCount, freeBeforeRebind);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

// ==================== ut_team_047 ====================
// 验证 BindSyncMem 后 HcommTeamDestroy 会释放 shadowMem
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_ShadowMemBound_Expect_ShadowMemFreed)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 4;
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_SUCCESS);
    // 确认 shadowMem 已分配
    {
        auto &mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto *entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        ASSERT_NE(entry->hostTeam.syncMem.shadowMem.addr, nullptr);
    }
    uint32_t freeBeforeDestroy = g_hrtFreeCallCount;

    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);
    // destroy 内 FreeTeamResources 会释放 shadowMem，free 计数应增加
    EXPECT_GT(g_hrtFreeCallCount, freeBeforeDestroy);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

// ==================== ut_team_048 ====================
// 验证首次 BindWindow 按 worldMemberNum 维度分配 mems
TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_FirstBind_Expect_MemsNumEqualsWorldMemberNum)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    std::vector<CommMem> mems(4);
    for (uint32_t i = 0; i < 4; i++) {
        mems[i].addr = malloc(1024);
        mems[i].size = 1024;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    HcommTeamWindowDesc desc{};
    desc.mems = mems.data();
    desc.memberNum = 4;
    ASSERT_EQ(HcommTeamWindowBindRemoteMems(team, handle, &desc), HCOMM_SUCCESS);

    auto &mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    auto *winEntry = mgr.FindWindowByHandleLocked(handle);
    ASSERT_NE(winEntry, nullptr);
    EXPECT_EQ(winEntry->hostWindow.memsNum, static_cast<uint64_t>(4));
    EXPECT_NE(winEntry->hostMems, nullptr);
    EXPECT_NE(winEntry->devMems, nullptr);

    for (uint32_t i = 0; i < 4; i++) {
        free(mems[i].addr);
    }
}

// ==================== ut_team_049 ====================
// 验证 BindWindow 累加合并：首次填部分槽，二次填其余槽，最终四槽都有值且未被覆盖
TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_MergeAcrossBinds_Expect_AllSlotsPreserved)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(team, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    // 首次 bind：填槽 0,1，槽 2,3 为空（addr=nullptr）
    std::vector<CommMem> mems1(4);
    void *addr0 = malloc(1024);
    void *addr1 = malloc(1024);
    mems1[0].addr = addr0;
    mems1[0].size = 1024;
    mems1[0].type = COMM_MEM_TYPE_DEVICE;
    mems1[1].addr = addr1;
    mems1[1].size = 1024;
    mems1[1].type = COMM_MEM_TYPE_DEVICE;
    HcommTeamWindowDesc desc1{};
    desc1.mems = mems1.data();
    desc1.memberNum = 4;
    ASSERT_EQ(HcommTeamWindowBindRemoteMems(team, handle, &desc1), HCOMM_SUCCESS);

    void *devMemsAfterFirst = nullptr;
    {
        auto &mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        auto *winEntry = mgr.FindWindowByHandleLocked(handle);
        ASSERT_NE(winEntry, nullptr);
        EXPECT_EQ(winEntry->hostMems[0].addr, addr0);
        EXPECT_EQ(winEntry->hostMems[1].addr, addr1);
        EXPECT_EQ(winEntry->hostMems[2].addr, nullptr); // 未填
        EXPECT_EQ(winEntry->hostMems[3].addr, nullptr); // 未填
        devMemsAfterFirst = winEntry->devMems;
    }

    // 二次 bind：填槽 2,3，槽 0,1 为空（addr=nullptr），不应覆盖已填的 0,1
    std::vector<CommMem> mems2(4);
    void *addr2 = malloc(1024);
    void *addr3 = malloc(1024);
    mems2[2].addr = addr2;
    mems2[2].size = 1024;
    mems2[2].type = COMM_MEM_TYPE_DEVICE;
    mems2[3].addr = addr3;
    mems2[3].size = 1024;
    mems2[3].type = COMM_MEM_TYPE_DEVICE;
    HcommTeamWindowDesc desc2{};
    desc2.mems = mems2.data();
    desc2.memberNum = 4;
    ASSERT_EQ(HcommTeamWindowBindRemoteMems(team, handle, &desc2), HCOMM_SUCCESS);

    {
        auto &mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        auto *winEntry = mgr.FindWindowByHandleLocked(handle);
        ASSERT_NE(winEntry, nullptr);
        // 累加合并：四槽都有值，首次的 0,1 未被覆盖
        EXPECT_EQ(winEntry->hostMems[0].addr, addr0);
        EXPECT_EQ(winEntry->hostMems[1].addr, addr1);
        EXPECT_EQ(winEntry->hostMems[2].addr, addr2);
        EXPECT_EQ(winEntry->hostMems[3].addr, addr3);
        EXPECT_EQ(winEntry->hostWindow.memsNum, static_cast<uint64_t>(4));
        // devMems 复用首次分配的同一块（不重新分配）
        EXPECT_EQ(winEntry->devMems, devMemsAfterFirst);
    }

    free(addr0);
    free(addr1);
    free(addr2);
    free(addr3);
}

// ==================== ut_team_050 ====================
// 验证跨 team bind 同一 window：worldTeam 与 subTeam 各自 bind 同一 window 均成功，memsNum 为 worldTeam 维度
TEST_F(TestHcommTeam, Ut_HcommTeamWindowBindRemoteMems_When_CrossTeamBind_Expect_BothSucceed)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, worldTeam, syncMemSize), HCOMM_SUCCESS);

    // subTeam：成员 {0,2}，selfMemberId=0
    std::vector<uint32_t> subIds = {0, 2};
    HcommTeamCreateDesc subDesc{};
    subDesc.memberNum = 2;
    subDesc.selfMemberId = 0;
    subDesc.worldMemberIds = subIds.data();
    subDesc.netLayer = 0;
    subDesc.requirement = {0, 0, 1, {}};
    HcommTeamHandle subTeam = nullptr;
    uint64_t subWsSize = 0;
    ASSERT_EQ(HcommTeamCreate(worldTeam, &subDesc, &subTeam, &subWsSize), HCOMM_SUCCESS);

    // window 注册到 worldTeam
    HcommWindowHandle handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(worldTeam, nullptr, &handle, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCOMM_SUCCESS);

    // worldTeam 首次 bind（worldMemberNum=4 维）
    std::vector<CommMem> worldMems(4);
    void *wAddr0 = malloc(1024);
    worldMems[0].addr = wAddr0;
    worldMems[0].size = 1024;
    worldMems[0].type = COMM_MEM_TYPE_DEVICE;
    HcommTeamWindowDesc worldDesc{};
    worldDesc.mems = worldMems.data();
    worldDesc.memberNum = 4;
    ASSERT_EQ(HcommTeamWindowBindRemoteMems(worldTeam, handle, &worldDesc), HCOMM_SUCCESS);

    // subTeam 二次 bind 同一 window（worldMemberNum=4 维，只填自己 member 的槽）
    std::vector<CommMem> subMems(4);
    void *sAddr2 = malloc(1024);
    subMems[2].addr = sAddr2; // subTeam 的 peer（world rank 2）对应 worldMemberId=2
    subMems[2].size = 1024;
    subMems[2].type = COMM_MEM_TYPE_DEVICE;
    HcommTeamWindowDesc subBindDesc{};
    subBindDesc.mems = subMems.data();
    subBindDesc.memberNum = 4;
    EXPECT_EQ(HcommTeamWindowBindRemoteMems(subTeam, handle, &subBindDesc), HCOMM_SUCCESS);

    {
        auto &mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        auto *winEntry = mgr.FindWindowByHandleLocked(handle);
        ASSERT_NE(winEntry, nullptr);
        EXPECT_EQ(winEntry->hostWindow.memsNum, static_cast<uint64_t>(4));
        // worldTeam 填的槽 0 + subTeam 填的槽 2 都在，累加合并
        EXPECT_EQ(winEntry->hostMems[0].addr, wAddr0);
        EXPECT_EQ(winEntry->hostMems[2].addr, sAddr2);
    }

    free(wAddr0);
    free(sAddr2);
}
