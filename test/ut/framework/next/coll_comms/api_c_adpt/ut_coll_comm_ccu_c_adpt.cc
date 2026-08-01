/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>

#define private public
#include "hccl_comm_pub.h"
#include "coll_comm.h"
#include "my_rank.h"
#include "ccu_instance_mgr.h"
#undef private

#include "ccu/ccu_res.h"
#include "hccl/hccl_ccu_res.h"

using namespace hccl;
using namespace hcomm;

class HcclCommAssignCcuInsTest : public testing::Test {
protected:
    void TearDown() override
    {
        CcuInstanceMgr::GetInstance(0).Deinit();
        GlobalMockObject::verify();
    }

    HcclComm BuildComm(bool isCommunicatorV2 = true, bool hasCollComm = true, bool hasMyRank = true,
        CcuInsHandle currentInsHandle = 0)
    {
        comm_ = std::make_unique<hcclComm>(0, 0, "assign_ccu_ut");
        comm_->devType_ = isCommunicatorV2 ? DevType::DEV_TYPE_950 : DevType::DEV_TYPE_COUNT;

        if (!hasCollComm) {
            return static_cast<HcclComm>(comm_.get());
        }

        comm_->collComm_ = std::make_unique<CollComm>(nullptr, 0, "assign_ccu_ut",
            ManagerCallbacks{}, CollCommInitMode::simpleMode);
        comm_->collComm_->deviceLogicId_ = 0;
        if (!hasMyRank) {
            return static_cast<HcclComm>(comm_.get());
        }

        myRank_ = std::make_shared<MyRank>(nullptr, 0, comm_->collComm_->config_,
            ManagerCallbacks{}, nullptr, nullptr);
        myRank_->ccuInsHandle_ = currentInsHandle;
        comm_->collComm_->myRank_ = myRank_;
        return static_cast<HcclComm>(comm_.get());
    }

    void MockCcuInstanceFound(CcuInsHandle insHandle = validInsHandle_)
    {
        auto &mgr = CcuInstanceMgr::GetInstance(0);
        mgr.initializedFlag_ = true;
        mgr.insMap_.emplace(insHandle, std::make_unique<CcuInstance>());
    }

    static constexpr CcuInsHandle validInsHandle_ = 100;
    std::unique_ptr<hcclComm> comm_;
    std::shared_ptr<MyRank> myRank_;
};

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_CommHasNoCcuInstance_Expect_ReturnIsHCCL_SUCCESS)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, 0);
    MockCcuInstanceFound();

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank_->GetCcuInstance(), insHandle);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_AssignSuccessAndCommDestroyed_Expect_DestroyCcuInstanceOnce)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, 0);
    std::weak_ptr<MyRank> myRankWeak = myRank_;
    MockCcuInstanceFound();
    MOCKER(HcommCcuInsDestroy)
        .expects(once())
        .with(insHandle)
        .will(returnValue(CcuResult::CCU_SUCCESS));

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(myRank_->GetCcuInstance(), insHandle);
    myRank_.reset();
    comm_.reset();
    EXPECT_TRUE(myRankWeak.expired());
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    constexpr CcuInsHandle insHandle = 1;

    HcclResult ret = HcclCommAssignCcuIns(nullptr, insHandle);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_InsHandleIsZero_Expect_ReturnIsHCCL_E_PARA)
{
    HcclComm comm = BuildComm();

    HcclResult ret = HcclCommAssignCcuIns(comm, 0);

    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(myRank_->GetCcuInstance(), 0);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_CommIsNotV2_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    constexpr CcuInsHandle insHandle = 1;
    HcclComm comm = BuildComm(false);

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(myRank_->GetCcuInstance(), 0);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_CollCommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    constexpr CcuInsHandle insHandle = 1;
    HcclComm comm = BuildComm(true, false);

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_MyRankIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    constexpr CcuInsHandle insHandle = 1;
    HcclComm comm = BuildComm(true, true, false);

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_InsHandleNotFound_Expect_ReturnIsHCCL_E_NOT_FOUND)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, 0);
    CcuInstanceMgr::GetInstance(0).Deinit();

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    EXPECT_EQ(myRank_->GetCcuInstance(), 0);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_CommHasDifferentCcuInstance_Expect_ReturnIsHCCL_E_PARA)
{
    constexpr CcuInsHandle oldInsHandle = validInsHandle_;
    constexpr CcuInsHandle newInsHandle = 200;
    HcclComm comm = BuildComm(true, true, true, oldInsHandle);
    MockCcuInstanceFound(newInsHandle);

    HcclResult ret = HcclCommAssignCcuIns(comm, newInsHandle);

    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(myRank_->GetCcuInstance(), oldInsHandle);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_CommHasSameCcuInstance_Expect_ReturnIsHCCL_E_PARA)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, insHandle);
    MockCcuInstanceFound();

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(myRank_->GetCcuInstance(), insHandle);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_InternalExceptionOccurs_Expect_ReturnIsHCCL_E_INTERNAL)
{
    constexpr CcuInsHandle insHandle = validInsHandle_;
    HcclComm comm = BuildComm(true, true, true, 0);

    MOCKER_CPP(&CcuInstanceMgr::Get)
        .expects(once())
        .with(insHandle)
        .will(throws(std::logic_error("test exception")));

    HcclResult ret = HcclCommAssignCcuIns(comm, insHandle);

    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(myRank_->GetCcuInstance(), 0);
}

TEST_F(HcclCommAssignCcuInsTest,
    Ut_HcclCommAssignCcuIns_When_TwoThreadsAssignConcurrently_Expect_OnlyOneSucceeds)
{
    constexpr CcuInsHandle firstInsHandle = validInsHandle_;
    constexpr CcuInsHandle secondInsHandle = 200;
    HcclComm comm = BuildComm(true, true, true, 0);
    MockCcuInstanceFound(firstInsHandle);
    MockCcuInstanceFound(secondInsHandle);

    std::atomic<uint32_t> readyCount{0};
    std::atomic<bool> start{false};
    HcclResult firstRet = HCCL_E_INTERNAL;
    HcclResult secondRet = HCCL_E_INTERNAL;
    auto assign = [&](CcuInsHandle insHandle, HcclResult &ret) {
        readyCount.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        ret = HcclCommAssignCcuIns(comm, insHandle);
    };

    std::thread firstThread(assign, firstInsHandle, std::ref(firstRet));
    std::thread secondThread(assign, secondInsHandle, std::ref(secondRet));
    while (readyCount.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    firstThread.join();
    secondThread.join();

    const bool firstSucceeded = firstRet == HCCL_SUCCESS && secondRet == HCCL_E_PARA;
    const bool secondSucceeded = firstRet == HCCL_E_PARA && secondRet == HCCL_SUCCESS;
    EXPECT_TRUE(firstSucceeded || secondSucceeded);
    EXPECT_EQ(myRank_->GetCcuInstance(), firstSucceeded ? firstInsHandle : secondInsHandle);
}
