/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <cstring>
#define private public
#include "local_cnt_notify.h"
#include "rdma_handle_manager.h"
#undef private
#include "orion_adapter_hccp.h"
using namespace Hccl;

class LocalCntNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LocalCntNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "LocalCntNotifyTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in LocalCntNotifyTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LocalCntNotifyTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
};

TEST_F(LocalCntNotifyTest, getExchangeDto_test)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_910A2));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByIndex).stubs().will(returnValue(static_cast<DevId>(1)));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    int a = 0;
    RtsCntNotify rtsCntNotify;
    RdmaHandleManager::GetInstance().tokenInfoMap[(void*)&a] = make_unique<TokenInfoManager>(0, (void*)&a);
    LocalCntNotify localCntNotify((void*)&a, &rtsCntNotify);

    localCntNotify.GetExchangeDto();
};
