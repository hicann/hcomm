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
#include <mockcpp/mockcpp.hpp>

#include "src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "ccu_comp.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

static void InitTestMissionReq(MissionReq& missionReq, MissionReqType reqType)
{
    missionReq.reqType = reqType;
    missionReq.req[0] = 0;
    missionReq.req[1] = 0;
}

class CcuResBatchAllocatorTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    static void TearDownTestCase() {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void SetUp() override {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void TearDown() override {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }

};

TEST_F(CcuResBatchAllocatorTest, Ut_Alloc_When_ReqTypeNotDefault_Expect_ForceSetToDefault)
{   
    hcomm::CcuResBatchAllocator allocat{};
    uintptr_t handleKey;
    MissionReq missionReq;
    MissionResInfo missionInfos;

    // 调用公共函数
    InitTestMissionReq(missionReq, MissionReqType::COMM_ENGINE_RESERVED);

    auto ret = allocat.missionMgr_.Alloc(handleKey, missionReq, missionInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_Alloc_When_ReqNumIs0_Expect_ReturnSuccessDirectly)
{
    hcomm::CcuResBatchAllocator allocat{};
    uintptr_t handleKey;
    MissionReq missionReq;
    MissionResInfo missionInfos;

    // 调用公共函数
    InitTestMissionReq(missionReq, MissionReqType::FUSION_MULTIPLE_DIE);

    auto ret = allocat.missionMgr_.Alloc(handleKey, missionReq, missionInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}