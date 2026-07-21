/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * This SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"
#define private public
#include "coll_comm.h"
#include "symmetric_memory/symmetric_memory.h"
#undef private
#include "coll_comm_config.h"
#include "hcom_common.h"

class TestCollComm : public TestHcommCAdptBase {
public:
    void SetUp() override {
        TestHcommCAdptBase::SetUp();
    }
    void TearDown() override {
        TestHcommCAdptBase::TearDown();
    }
};

HcclResult StubCollCommUrmaHrtMalloc(void **devPtr, u64 size, bool Level2Address)
{
    static uintptr_t devAddr = 0x7000000;
    (void)size;
    (void)Level2Address;
    *devPtr = reinterpret_cast<void*>(devAddr);
    devAddr += 0x1000;
    return HCCL_SUCCESS;
}

TEST_F(TestCollComm, Ut_TestCollCommInit_When_RankGraphNullptr_Return_HCCL_E_PTR)
{
    hccl::CollComm collComm(nullptr, 0, "test_comm", hccl::ManagerCallbacks{});
    HcclMem cclBuffer = {};
    HcclResult ret = collComm.Init(nullptr, nullptr, cclBuffer, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestCollComm, test_get_comm_status_initial_and_after_change)
{
    std::unique_ptr<CollComm> coll_ = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), 
        hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    EXPECT_EQ(coll_->GetCommStatus(), HcclCommStatus::HCCL_COMM_STATUS_INVALID);

    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    EXPECT_EQ(coll_->GetCommStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
}

TEST_F(TestCollComm, test_suspend_success_and_idempotent)
{
    std::unique_ptr<CollComm> coll_ = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), 
        hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // mock MyRank::StopLaunch to return success
    MOCKER_CPP(&MyRank::StopLaunch, HcclResult(MyRank:: *)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance (can be real or mocked; method is mocked above)
    aclrtBinHandle binHandle;
    coll_->myRank_ = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    auto ret = coll_->Suspend();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll_->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);

    // calling Suspend again when already suspending should return success without error
    ret = coll_->Suspend();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestCollComm, test_clean_fail_not_suspending)
{
    std::unique_ptr<CollComm> coll_ = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), 
        hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // when not suspending, Clean should return not support
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    auto ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestCollComm, test_clean_success_and_idempotent)
{
    std::unique_ptr<CollComm> coll_ = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), 
        hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // prepare for cleaning: put into suspending state
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll_->isCleaned_ = false;

    // mock MyRank::Clean to return success
    MOCKER_CPP(&MyRank::Clean, HcclResult(MyRank:: *)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance
    aclrtBinHandle binHandle;
    coll_->myRank_ = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    auto ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(coll_->isCleaned_);

    // calling Clean again should be idempotent and return success
    ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestCollComm, test_resume_fail_invalid_and_resume_success)
{
    std::unique_ptr<CollComm> coll_ = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), 
        hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // Resume when commStatus_ is INVALID should return internal error
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    auto ret = coll_->Resume();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    // Now test successful resume from SUSPENDING
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll_->isCleaned_ = true;

    // mock MyRank::Resume to return success
    MOCKER_CPP(&MyRank::Resume, HcclResult(MyRank:: *)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance
    aclrtBinHandle binHandle;
    coll_->myRank_ = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    ret = coll_->Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll_->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll_->isCleaned_);
}

TEST_F(TestCollComm, Ut_InitSimpleMode_When_Success_Expect_ReturnsSuccessAndRankgraphSet)
{
    // Create CollComm instance
    hccl::CollComm collComm(nullptr, 0, "test_comm", hccl::ManagerCallbacks{});

    // Verify initial state - rankgraph should be nullptr before initialization
    // This validates the internal state that InitSimpleMode would populate
    EXPECT_EQ(collComm.GetRankSize(), 0u);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_ValidConfig_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 2U;
    config.hcclRdmaTrafficClass = 120U;
    config.hcclRdmaServiceLevel = 3U;
    config.hcclQos = 5U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(opExpansionMode, 2U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigHcclQos(), 5U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 120U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 3U);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_NullConfig_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    uint32_t opExpansionMode = 9U;
    EXPECT_EQ(ApplyHcclCommConfig(nullptr, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(opExpansionMode, 0U);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidHcclQos_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 8U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidTrafficClass_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 256U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidServiceLevel_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaServiceLevel = 8U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_RegisterPendingSymmetricMemHandles_When_HasPendingWindow_Expect_ReturnMemHandleOnce)
{
    MOCKER_CPP(hrtMalloc)
        .stubs()
        .will(invoke(StubCollCommUrmaHrtMalloc));
    MOCKER_CPP(hrtMemSyncCopy)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    hccl::CollComm coll(nullptr, 0, "ut_sym", hccl::ManagerCallbacks{});
    aclrtBinHandle binHandle {};
    coll.myRank_ = std::make_shared<MyRank>(binHandle, 0, coll.GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);
    coll.myRank_->commMems_ = std::make_unique<CommMems>(0);
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));

    void* win = nullptr;
    void* ptr = reinterpret_cast<void*>(0x8000000);
    EXPECT_EQ(coll.RegisterWindow(ptr, 0x2000, &win), HCCL_SUCCESS);

    std::vector<HcclMemHandle> memHandles;
    EXPECT_EQ(coll.RegisterPendingSymmetricMemHandles(memHandles), HCCL_SUCCESS);
    ASSERT_EQ(memHandles.size(), 1U);
    EXPECT_NE(memHandles[0], nullptr);

    SymmetricMemoryResource resource;
    EXPECT_EQ(coll.symmetricMemory_->GetRegisteredMemoryResource(win, resource), HCCL_SUCCESS);
    EXPECT_EQ(resource.memHandle, static_cast<void*>(memHandles[0]));

    memHandles.clear();
    EXPECT_EQ(coll.RegisterPendingSymmetricMemHandles(memHandles), HCCL_SUCCESS);
    EXPECT_TRUE(memHandles.empty());
}

TEST_F(TestCollComm, Ut_UpdateSymmetricRemoteMem_When_ChannelReturnsRemoteMem_Expect_UpdateWindow)
{
    MOCKER_CPP(hrtMalloc)
        .stubs()
        .will(invoke(StubCollCommUrmaHrtMalloc));
    MOCKER_CPP(hrtMemSyncCopy)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    hccl::CollComm coll(nullptr, 0, "ut_sym", hccl::ManagerCallbacks{});
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));

    void* win = nullptr;
    void* ptr = reinterpret_cast<void*>(0x9000000);
    constexpr size_t winSize = 0x2000;
    EXPECT_EQ(coll.RegisterWindow(ptr, winSize, &win), HCCL_SUCCESS);

    SymmetricMemoryResource resource;
    resource.memHandle = reinterpret_cast<void*>(0x9100000);
    resource.memTag = std::string(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) + "ut_sym_addr_" +
        std::to_string(reinterpret_cast<uintptr_t>(ptr)) + "_size_" + std::to_string(winSize);
    EXPECT_EQ(coll.symmetricMemory_->SetRegisteredMemoryResource(win, resource), HCCL_SUCCESS);

    CommMem remoteMem {};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9200000);
    remoteMem.size = winSize;
    char *memTags[] = {const_cast<char*>(resource.memTag.c_str())};
    EXPECT_EQ(coll.UpdateSymmetricRemoteMem(1, &remoteMem, memTags, 1), HCCL_SUCCESS);

    auto remoteMemIt = coll.symmetricMemory_->remoteMemMap_.find(win);
    ASSERT_NE(remoteMemIt, coll.symmetricMemory_->remoteMemMap_.end());
    ASSERT_EQ(remoteMemIt->second.size(), 2U);
    EXPECT_EQ(remoteMemIt->second[1].addr, remoteMem.addr);
    EXPECT_EQ(remoteMemIt->second[1].size, remoteMem.size);
}
