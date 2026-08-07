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
#define private public
#include "ipc_local_notify.h"
#include "rdma_local_notify.h"
#include "ub_local_notify.h"
#include "rdma_handle_manager.h"
#undef private
#include "orion_adapter_hccp.h"
#include "hccp_ctx.h"

using namespace Hccl;

class IpcLocalNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IpcLocalNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "IpcLocalNotifyTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
        MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
        MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));
        MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));

        std::cout << "A Test case in IpcLocalNotifyTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in IpcLocalNotifyTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    s32 fakePid = 100;
    char fakeName[RTS_IPC_MEM_NAME_LEN] = "testRtsNotify";
};

TEST_F(IpcLocalNotifyTest, ipc_local_rtsNotify_post_wait_grant_describe)
{
    Stream stream;
    IpcLocalNotify ipcLocalNotify;
    ipcLocalNotify.Grant(100);
    ipcLocalNotify.Describe();

    ipcLocalNotify.Wait(stream, 100);
    ipcLocalNotify.Post(stream);

    GlobalMockObject::verify();
}

class RdmaLocalNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RdmaLocalNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RdmaLocalNotifyTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
        MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
        MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));
        MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
        MOCKER(HrtNotifyCreateWithFlag).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(notifyInfo));

        MOCKER(HrtRaGetNotifyBaseAddr)
            .stubs()
            .with(mockcpp::any(), outBoundP(&fakeVa, sizeof(fakeVa)), outBoundP(&fakeSize, sizeof(fakeSize)))
            .will(returnValue(1));

        std::cout << "A Test case in RdmaLocalNotifyTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RdmaLocalNotifyTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u32 fakeOffset = 200;
    u64 fakeAddress = 300;
    s32 fakePid = 100;
    u64 fakeVa = 100;
    u64 fakeSize = 1024;
    char fakeName[RTS_IPC_MEM_NAME_LEN] = "testRtsNotify";
    RtNotify_t notifyInfo;
};

TEST_F(RdmaLocalNotifyTest, rdma_local_notify_describe_wait_post)
{
    Stream stream;
    RdmaHandle rdmaHandle = (void*)0x100;
    RdmaLocalNotify rdmaLocalNotify(rdmaHandle);

    cout << rdmaLocalNotify.Describe() << endl;

    rdmaLocalNotify.Wait(stream, 100);

    EXPECT_THROW(rdmaLocalNotify.Post(stream), NotSupportException);
}

class UbLocalNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UbLocalNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "UbLocalNotifyTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        localMemRegInfo.handle = fakeMemHandle;
        memcpy_s(localMemRegInfo.key, HRT_UB_MEM_KEY_MAX_LEN, fakeKey, HRT_UB_MEM_KEY_MAX_LEN);
        localMemRegInfo.tokenId = fakeTokenId;
        localMemRegInfo.targetSegVa = fakeSegVa;
        HrtDevResAddrInfo resAddrInfo;
        MOCKER(HrtGetDevResAddress).stubs().with(mockcpp::any()).will(returnValue(resAddrInfo));

        RequestHandle fakeReqHandle = 1;

        vector<char_t> out;
        out.resize(sizeof(struct MrRegInfoT));
        struct MrRegInfoT* info = reinterpret_cast<struct MrRegInfoT*>(out.data());
        memcpy_s(info->out.key.value, HRT_UB_MEM_KEY_MAX_LEN, fakeKey, HRT_UB_MEM_KEY_MAX_LEN);
        info->out.key.size = HRT_UB_MEM_KEY_MAX_LEN;
        info->out.ub.tokenId = fakeTokenId;
        info->out.ub.targetSegHandle = fakeSegVa;

        MOCKER(RaUbLocalMemRegAsync)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), outBound(out), outBound(reinterpret_cast<void*>(fakeMemHandle)))
            .will(returnValue(fakeReqHandle));
        MOCKER(RaUbLocalMemUnregAsync).stubs().will(returnValue(fakeReqHandle));

        MOCKER(HrtReleaseDevResAddress).stubs().with(mockcpp::any());

        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
        MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
        MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));
        MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
        MOCKER(HrtRaGetNotifyBaseAddr)
            .stubs()
            .with(mockcpp::any(), outBoundP(&fakeVa, sizeof(fakeVa)), outBoundP(&fakeSize, sizeof(fakeSize)))
            .will(returnValue(1));

        std::cout << "A Test case in UbLocalNotifyTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in UbLocalNotifyTest TearDown" << std::endl;
    }

    HrtRaUbLocalMemRegOutParam localMemRegInfo;
    u32 fakeMemAddr = 0;
    u32 fakeMemSize = 0;
    u64 fakeSegVa = 0x200;
    u8 fakeKey[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 fakeTokenId = 1;
    u64 fakeMemHandle = 0x200;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    s32 fakePid = 100;
    u64 fakeVa = 100;
    u64 fakeSize = 1024;
    char fakeName[RTS_IPC_MEM_NAME_LEN] = "testRtsNotify";
};

TEST_F(UbLocalNotifyTest, ub_local_notify_initialize)
{
    // given
    RdmaHandle rdmaHandle = (void*)0x100;
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    Stream stream;

    pair<u64, u32> notifyInfoPair(fakeMemHandle, 0);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(notifyInfoPair));
    HrtRaUbLocalMemRegOutParam hrtRaUbLocalMemRegOutParam;
    hrtRaUbLocalMemRegOutParam.handle = fakeMemHandle;
    MOCKER(HrtRaUbLocalMemReg).stubs().will(returnValue(hrtRaUbLocalMemRegOutParam));

    // when
    UbLocalNotify ubLocalNotify(rdmaHandle);

    ubLocalNotify.Wait(stream, 100);

    // then
    EXPECT_EQ(ubLocalNotify.addr, 0);
    EXPECT_EQ(ubLocalNotify.size, ubLocalNotify.size);
    EXPECT_EQ(ubLocalNotify.tokenId, localMemRegInfo.tokenId >> 8);
    EXPECT_EQ(ubLocalNotify.memHandle, localMemRegInfo.handle);
    EXPECT_EQ(memcmp(ubLocalNotify.key, localMemRegInfo.key, HRT_UB_MEM_KEY_MAX_LEN), 0);
}

TEST_F(UbLocalNotifyTest, ub_local_notify_does_not_support_post)
{
    // given
    RdmaHandle rdmaHandle = (void*)0x100;
    UbLocalNotify ubLocalNotify(rdmaHandle);
    Stream stream;

    // then
    EXPECT_THROW(ubLocalNotify.Post(stream), NotSupportException);
}

TEST_F(UbLocalNotifyTest, getExchangeDto_test)
{
    RdmaHandle rdmaHandle = (void*)0x100;
    UbLocalNotify ubLocalNotify(rdmaHandle);

    ubLocalNotify.GetExchangeDto();
};
