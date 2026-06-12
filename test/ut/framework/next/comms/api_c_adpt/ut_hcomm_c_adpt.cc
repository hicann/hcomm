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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "channel_process.h"
#include "endpoint_map.h"

using namespace hcomm;

HcclResult StubServerSocketGetListenPort(Endpoint* /*endpoint*/, uint32_t* port)
{
    *port = 12345;
    return HCCL_SUCCESS;
}

class HcommCAdptTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HcommCAdptTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HcommCAdptTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in HcommCAdptTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcommCAdptTest TearDown" << std::endl;
    }
};

TEST_F(HcommCAdptTest, ut_HcommChannelGet_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    void* channel = nullptr;
    MOCKER(ChannelProcess::ChannelGet)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGet(channelHandle, &channel);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_Normal_Expect_Success)
{
    ChannelHandle channelList[2] = {
        0x12345,
        0x12346
    };
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], 0);
    EXPECT_EQ(statusList[1], 0);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_ChannelListNull_Expect_E_PTR)
{
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(nullptr, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_StatusListNull_Expect_E_PTR)
{
    ChannelHandle channelList[2] = {
        0x12345,
        0x12346
    };
    HcommResult ret = HcommChannelGetStatus(channelList, 2, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_ListNumZero_Expect_E_PARA)
{
    ChannelHandle channelList[2] = {
        0x12345,
        0x12346
    };
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(channelList, 0, statusList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetNotifyNum_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    uint32_t notifyNum = 0;
    MOCKER(ChannelProcess::ChannelGetNotifyNum)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGetNotifyNum(channelHandle, &notifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelDestroy_When_Normal_Expect_Success)
{
    ChannelHandle channels[2] = {
        0x12345,
        0x12346
    };
    MOCKER(ChannelProcess::ChannelDestroy)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelDestroy(channels, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetRemoteMem_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char* memTagsStorage[2] = {nullptr, nullptr};
    char** memTags = memTagsStorage;
    MOCKER(ChannelProcess::ChannelGetRemoteMem)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGetRemoteMem(channelHandle, &remoteMem, &memNum, memTags);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetRemoteMems_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    CommMem* remoteMems = nullptr;
    char** memTags = nullptr;
    uint32_t memNum = 0;
    MOCKER(ChannelProcess::ChannelGetUserRemoteMem)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGetRemoteMems(channelHandle, &memNum, &remoteMems, &memTags);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_When_Normal_Expect_Success)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_When_ChannelDescsNull_Expect_E_PTR)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_When_ChannelNumZero_Expect_E_PARA)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, &channelDesc, 0, channels);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_When_ChannelDescsNull_Expect_E_PTR)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_When_ChannelNumZero_Expect_E_PARA)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, &channelDesc, 0, channels);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGet_When_NotFound_Expect_E_NOT_FOUND)
{
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x12345678);
    void* endpoint = nullptr;
    HcommResult ret = HcommEndpointGet(handle, &endpoint);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGet_When_EndpointPtrNull_Expect_E_PTR)
{
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x12345678);
    HcommResult ret = HcommEndpointGet(handle, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_When_NotAiCpu_Expect_Success)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(ChannelProcess::ConnectChannels)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(ChannelProcess::SaveChannels)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommEngineCtxCopy_When_CPU_Expect_Success)
{
    void* dstCtx = malloc(1024);
    void* srcCtx = malloc(1024);
    ASSERT_NE(dstCtx, nullptr);
    ASSERT_NE(srcCtx, nullptr);
    memset(dstCtx, 0, 1024);
    memset(srcCtx, 1, 1024);
    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CPU, dstCtx, srcCtx, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    free(dstCtx);
    free(srcCtx);
}

TEST_F(HcommCAdptTest, ut_HcommEngineCtxCopy_When_CPU_TS_Expect_Success)
{
    void* dstCtx = malloc(1024);
    void* srcCtx = malloc(1024);
    ASSERT_NE(dstCtx, nullptr);
    ASSERT_NE(srcCtx, nullptr);
    memset(dstCtx, 0, 1024);
    memset(srcCtx, 1, 1024);
    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CPU_TS, dstCtx, srcCtx, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    free(dstCtx);
    free(srcCtx);
}

TEST_F(HcommCAdptTest, ut_HcommEngineCtxCopy_When_CCU_Expect_Success)
{
    void* dstCtx = malloc(1024);
    void* srcCtx = malloc(1024);
    ASSERT_NE(dstCtx, nullptr);
    ASSERT_NE(srcCtx, nullptr);
    memset(dstCtx, 0, 1024);
    memset(srcCtx, 1, 1024);
    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CCU, dstCtx, srcCtx, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    free(dstCtx);
    free(srcCtx);
}

TEST_F(HcommCAdptTest, ut_HcommChannelDescInit_When_Normal_Expect_Success)
{
    HcommChannelDesc channelDesc{};
    HcommResult ret = HcommChannelDescInit(&channelDesc, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_AICPU_Expect_LoadKernel)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(ChannelProcess::ConnectChannels)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(ChannelProcess::SaveChannels)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_CPU_Expect_Success)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_CCU_Expect_Success)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CCU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommResMgrInit_When_Normal_Expect_Success)
{
    HcommResult ret = HcommResMgrInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommResMgrInit_MultiDevice_Expect_Success)
{
    HcommResult ret1 = HcommResMgrInit();
    HcommResult ret2 = HcommResMgrInit();
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_PortNull_Expect_E_PTR)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_HandleInvalid_Expect_E_NOT_FOUND)
{
    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(nullptr, &port);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

class StubEndpoint final : public Endpoint {
public:
    explicit StubEndpoint() : Endpoint(EndpointDesc{})
    {
    }
    HcclResult Init() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult ServerSocketListen(const uint32_t port) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult RegisterMemory(HcommMem mem, const char *memTag, void **memHandle) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult UnregisterMemory(void *memHandle) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult MemoryExport(void *memHandle, void **memDesc, uint32_t *memDescLen) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult MemoryImport(const void *memDesc, uint32_t descLen, HcommMem *outMem) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult MemoryUnimport(const void *memDesc, uint32_t descLen) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult GetAllMemHandles(void **memHandles, uint32_t *memHandleNum) override
    {
        return HCCL_SUCCESS;
    }
};

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_ServerSocketNotSupport_Expect_E_NOT_SUPPORT)
{
    uint32_t port = 0;
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    StubEndpoint stubEndpoint;

    MOCKER_CPP(&HcommEndpointMap::GetEndpoint, Endpoint * (HcommEndpointMap::*)(EndpointHandle))
        .stubs()
        .will(returnValue(static_cast<Endpoint *>(&stubEndpoint)));

    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, &port);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}