/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "coll_comm_mgr.h"
#include "hcomm_c_adpt.h"
#include "hcomm_result_defs.h"

extern thread_local s32 g_hcclDeviceId;

// InitBaseCommRes 为 HcommResMgrInit 的透传封装，捕获传入 devId 以断言透传正确
static u32 g_ut_capturedHcommResMgrInitDevId = 0xFFFFFFFFu;
static HcommResult stub_CollCommMgrTest_HcommResMgrInit(uint32_t devPhyId)
{
    g_ut_capturedHcommResMgrInitDevId = devPhyId;
    return static_cast<HcommResult>(HCOMM_SUCCESS);
}

class CollCommMgrTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // Reset all opHcomInfos_ slots to clean state
        for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; i++) {
            CollCommMgr::GetInstance().LegacyGetOpHcomInfo(i).isUsed = false;
        }
        g_hcclDeviceId = INVALID_INT;
    }

    void TearDown() override
    {
        // Clean up all slots
        for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; i++) {
            CollCommMgr::GetInstance().LegacyGetOpHcomInfo(i).isUsed = false;
        }
        g_hcclDeviceId = INVALID_INT;
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

// ===== LegacyGetOpHcomInfo =====

TEST_F(CollCommMgrTest, Ut_LegacyGetOpHcomInfo_When_DevIdNormal_Expect_ReturnCorrespondingSlot)
{
    u32 devId = 3;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(devId);
    HcclOpInfoCtx& expectedSlot = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(devId);
    EXPECT_EQ(&info, &expectedSlot);
    EXPECT_EQ(info.isUsed, false);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetOpHcomInfo_When_DevIdOutOfRange_Expect_ClampToBackupSlot)
{
    u32 outOfRangeDevId = MAX_MODULE_DEVICE_NUM + 10;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(outOfRangeDevId);
    HcclOpInfoCtx& backupSlot = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM);
    EXPECT_EQ(&info, &backupSlot);
}

// ===== LegacyGetHcclExistDeviceOpInfoCtx =====

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclExistDeviceOpInfoCtx_When_SlotAlreadyUsed_Expect_DirectReturnDevIdUnchanged)
{
    s32 devId = 2;
    CollCommMgr::GetInstance().LegacyGetOpHcomInfo(devId).isUsed = true;
    s32 originalDevId = devId;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclExistDeviceOpInfoCtx(devId);
    EXPECT_EQ(devId, originalDevId);
    EXPECT_EQ(info.isUsed, true);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclExistDeviceOpInfoCtx_When_SlotNotUsedAndBackupUsed_Expect_FallbackToBackup)
{
    s32 devId = 3;
    CollCommMgr::GetInstance().LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM).isUsed = true;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclExistDeviceOpInfoCtx(devId);
    EXPECT_EQ(devId, static_cast<s32>(MAX_MODULE_DEVICE_NUM));
    EXPECT_EQ(info.isUsed, true);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclExistDeviceOpInfoCtx_When_SlotNotUsedAndBackupNotUsed_Expect_MarkAndReturn)
{
    s32 devId = 4;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclExistDeviceOpInfoCtx(devId);
    EXPECT_EQ(devId, 4);
    EXPECT_EQ(info.isUsed, true);
}

// ===== LegacyGetHcclOpInfoCtx =====

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclOpInfoCtx_When_HcclGetDeviceIdSuccess_Expect_DelegateToExistDevice)
{
    g_hcclDeviceId = 0;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    EXPECT_EQ(g_hcclDeviceId, 0);
    EXPECT_EQ(info.isUsed, true);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclOpInfoCtx_When_HcclGetDeviceIdFailAndHasUsedSlot_Expect_ScanFirstUsed)
{
    g_hcclDeviceId = INVALID_INT;
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    CollCommMgr::GetInstance().LegacyGetOpHcomInfo(5).isUsed = true;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    EXPECT_EQ(g_hcclDeviceId, 5);
    EXPECT_EQ(info.isUsed, true);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclOpInfoCtx_When_HcclGetDeviceIdFailAndNoUsedSlot_Expect_FallbackToBackup)
{
    g_hcclDeviceId = INVALID_INT;
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    EXPECT_EQ(g_hcclDeviceId, static_cast<s32>(MAX_MODULE_DEVICE_NUM));
    EXPECT_EQ(info.isUsed, true);
}

// ===== InitBaseCommRes =====
// InitBaseCommRes 仅透传调用 HcommResMgrInit(devId)，验证其以传入 devId 调用底层入口。
// 使用 stubs + invoke 捕获参数，避免 expects(once) 受 TearDown 链路额外调用影响。

TEST_F(CollCommMgrTest, Ut_InitBaseCommRes_When_Called_Expect_InvokeHcommResMgrInitWithDevId)
{
    const u32 devId = 7;
    g_ut_capturedHcommResMgrInitDevId = 0xFFFFFFFFu;
    MOCKER(HcommResMgrInit).stubs().will(invoke(stub_CollCommMgrTest_HcommResMgrInit));
    CollCommMgr::GetInstance().InitBaseCommRes(devId);
    EXPECT_EQ(g_ut_capturedHcommResMgrInitDevId, devId);
}

// ===== RegisteCollComm / UnRegisteCollComm =====
// taskAbortHandler_ 收编进 CollCommMgr 后，注册/注销为统一入口，同时维护 allCollComms_ 与 taskAbortHandler_。
// 使用 simpleMode 轻量桩验证 map 状态与往返安全：simpleMode 桩构造不解引用 null comm，析构短路返回；
// taskAbortHandler Register/UnRegister 全程仅存指针/指针比较；UnRegisterToClusterMonitor 在 initialized_==false 下
// no-op。

TEST_F(CollCommMgrTest, Ut_RegisteCollComm_When_NewComm_Expect_AddedToMap)
{
    const std::string commId = "ut_reg_stub";
    CollComm stub(nullptr, 0, commId, ManagerCallbacks{}, CollCommInitMode::simpleMode);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
    CollCommMgr::GetInstance().RegisteCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 1U);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().at(commId), &stub);
    CollCommMgr::GetInstance().UnRegisteCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
}

TEST_F(CollCommMgrTest, Ut_RegisteCollComm_When_DuplicateRegister_Expect_OverwriteNotGrow)
{
    const std::string commId = "ut_reg_dup";
    CollComm stub(nullptr, 0, commId, ManagerCallbacks{}, CollCommInitMode::simpleMode);
    CollCommMgr::GetInstance().RegisteCollComm(&stub);
    CollCommMgr::GetInstance().RegisteCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 1U);
    CollCommMgr::GetInstance().UnRegisteCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
}

TEST_F(CollCommMgrTest, Ut_UnRegisteCollComm_When_NotRegistered_Expect_NoCrashNoChange)
{
    const std::string commId = "ut_unreg_noreg";
    CollComm stub(nullptr, 0, commId, ManagerCallbacks{}, CollCommInitMode::simpleMode);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
    // 未注册直接注销：map erase no-op，taskAbortHandler UnRegister 仅告警，clusterMonitor initialized_==false no-op
    CollCommMgr::GetInstance().UnRegisteCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
}
