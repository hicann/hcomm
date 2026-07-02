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
#include <cstddef>
#include <cstdlib>
#include <mockcpp/mockcpp.hpp>
#include <string>
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "channel_process.h"
#include "endpoint_map.h"
#include "env_config/env_config.h"
#include "nic_plugin_dispatcher.h"
#include "nic_plugin_manager.h"
#include "orion_adpt_utils.h"
#include "hccp_peer_manager.h"
#include "rdma_handle_manager.h"
#include "hccp_nda.h"
#include "adapter_rts_common.h"

using namespace hcomm;

namespace {
struct FakePluginChannelState {
    uint32_t getStatusCalls = 0;
};

struct FakePluginEndpointState {
    bool inited = false;
};

struct FakePluginScenario {
    uint32_t readyAfterCalls = 1;
    HcommResult getStatusRet = HCCL_SUCCESS;
    uint32_t createChannelCalls = 0;
    uint32_t destroyChannelCalls = 0;
    uint32_t createEndpointCalls = 0;
    uint32_t initEndpointCalls = 0;
    uint32_t destroyEndpointCalls = 0;
    uint32_t registerMemoryCalls = 0;
    uint32_t unregisterMemoryCalls = 0;
    uint32_t memoryExportCalls = 0;
    uint32_t memoryImportCalls = 0;
    uint32_t memoryUnimportCalls = 0;
    uint32_t getUserRemoteMemCalls = 0;
    HcommMemHandle lastMemHandle = nullptr;
    const void *lastMemDesc = nullptr;
    uint32_t lastMemDescLen = 0;
};

FakePluginScenario g_fakePluginScenario;
constexpr uintptr_t FAKE_PLUGIN_MEM_HANDLE = 0x12345678U;
char g_fakePluginMemDesc[8] = {};
char g_fakeRemoteMem0[0x100] = {};
char g_fakeRemoteMem1[0x200] = {};
CommMem g_fakeRemoteMems[2] = {
    {COMM_MEM_TYPE_HOST, g_fakeRemoteMem0, sizeof(g_fakeRemoteMem0)},
    {COMM_MEM_TYPE_HOST, g_fakeRemoteMem1, sizeof(g_fakeRemoteMem1)}
};
char g_fakeRemoteTag0[] = "fake_remote_0";
char g_fakeRemoteTag1[] = "fake_remote_1";
char *g_fakeRemoteTags[2] = {g_fakeRemoteTag0, g_fakeRemoteTag1};

int32_t FakePluginCreateEndpoint(const EndpointDesc *endpointDesc, void **outCtx, HcommNicEndpointOps **outOps);
int32_t FakePluginEndpointInit(void *ctx);
void FakePluginEndpointDestroy(void *ctx);
int32_t FakePluginRegisterMemory(void *ctx, const CommMem *mem, const char *tag, void **handle);
int32_t FakePluginUnregisterMemory(void *ctx, void *handle);
int32_t FakePluginMemoryExport(void *ctx, void *handle, void **desc, uint32_t *descLen);
int32_t FakePluginMemoryImport(void *ctx, const void *desc, uint32_t descLen, CommMem *outMem);
int32_t FakePluginMemoryUnimport(void *ctx, const void *desc, uint32_t descLen);
int32_t FakePluginCreateChannel(void *epCtx, const HcommChannelDesc *channelDesc,
    void **outCtx, HcommNicChannelOps **outOps);
int32_t FakePluginGetStatus(void *ctx, int32_t *status);
int32_t FakePluginGetUserRemoteMem(void *ctx, CommMem **mem, char ***tags, uint32_t *num);
void FakePluginDestroyChannel(void *ctx);

HcommNicEndpointOps g_fakeEndpointOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    FakePluginEndpointInit,
    FakePluginRegisterMemory,
    FakePluginUnregisterMemory,
    FakePluginMemoryExport,
    FakePluginMemoryImport,
    FakePluginMemoryUnimport,
    FakePluginEndpointDestroy
};

HcommNicEndpointOps g_fakeUnsupportedEndpointOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0}
};

HcommNicChannelOps g_fakeChannelOps = {
    {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, sizeof(HcommNicChannelOps), 0},
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    FakePluginDestroyChannel,
    FakePluginGetStatus,
    nullptr,
    FakePluginGetUserRemoteMem
};

HcommNicPluginInfo g_fakePluginInfo = {
    {HCOMM_NIC_PLUGIN_INFO_VERSION, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD, sizeof(HcommNicPluginInfo), 0},
    "fake_plugin",
    1,
    {COMM_PROTOCOL_ROCE}
};

NicPluginEntry g_fakePluginEntry = {
    nullptr,
    &g_fakePluginInfo,
    FakePluginCreateEndpoint,
    FakePluginCreateChannel
};

int32_t FakePluginCreateEndpoint(const EndpointDesc *endpointDesc, void **outCtx, HcommNicEndpointOps **outOps)
{
    EXPECT_NE(endpointDesc, nullptr);
    g_fakePluginScenario.createEndpointCalls++;
    *outCtx = new FakePluginEndpointState();
    *outOps = &g_fakeEndpointOps;
    return HCCL_SUCCESS;
}

int32_t FakePluginEndpointInit(void *ctx)
{
    g_fakePluginScenario.initEndpointCalls++;
    auto *state = static_cast<FakePluginEndpointState *>(ctx);
    state->inited = true;
    return HCCL_SUCCESS;
}

void FakePluginEndpointDestroy(void *ctx)
{
    g_fakePluginScenario.destroyEndpointCalls++;
    delete static_cast<FakePluginEndpointState *>(ctx);
}

int32_t FakePluginRegisterMemory(void *ctx, const CommMem *mem, const char *tag, void **handle)
{
    (void)ctx;
    (void)mem;
    (void)tag;
    g_fakePluginScenario.registerMemoryCalls++;
    *handle = reinterpret_cast<void *>(FAKE_PLUGIN_MEM_HANDLE);
    g_fakePluginScenario.lastMemHandle = *handle;
    return HCCL_SUCCESS;
}

int32_t FakePluginUnregisterMemory(void *ctx, void *handle)
{
    (void)ctx;
    g_fakePluginScenario.unregisterMemoryCalls++;
    g_fakePluginScenario.lastMemHandle = handle;
    return HCCL_SUCCESS;
}

int32_t FakePluginMemoryExport(void *ctx, void *handle, void **desc, uint32_t *descLen)
{
    (void)ctx;
    g_fakePluginScenario.memoryExportCalls++;
    g_fakePluginScenario.lastMemHandle = handle;
    *desc = g_fakePluginMemDesc;
    *descLen = sizeof(g_fakePluginMemDesc);
    return HCCL_SUCCESS;
}

int32_t FakePluginMemoryImport(void *ctx, const void *desc, uint32_t descLen, CommMem *outMem)
{
    (void)ctx;
    g_fakePluginScenario.memoryImportCalls++;
    g_fakePluginScenario.lastMemDesc = desc;
    g_fakePluginScenario.lastMemDescLen = descLen;
    outMem->type = COMM_MEM_TYPE_HOST;
    outMem->addr = const_cast<void *>(desc);
    outMem->size = descLen;
    return HCCL_SUCCESS;
}

int32_t FakePluginMemoryUnimport(void *ctx, const void *desc, uint32_t descLen)
{
    (void)ctx;
    g_fakePluginScenario.memoryUnimportCalls++;
    g_fakePluginScenario.lastMemDesc = desc;
    g_fakePluginScenario.lastMemDescLen = descLen;
    return HCCL_SUCCESS;
}

int32_t FakePluginCreateChannel(void *epCtx, const HcommChannelDesc *channelDesc,
    void **outCtx, HcommNicChannelOps **outOps)
{
    (void)epCtx;
    EXPECT_NE(channelDesc, nullptr);
    g_fakePluginScenario.createChannelCalls++;
    *outCtx = new FakePluginChannelState();
    *outOps = &g_fakeChannelOps;
    return HCCL_SUCCESS;
}

int32_t FakePluginGetStatus(void *ctx, int32_t *status)
{
    if (g_fakePluginScenario.getStatusRet != HCCL_SUCCESS) {
        return g_fakePluginScenario.getStatusRet;
    }
    auto *state = static_cast<FakePluginChannelState *>(ctx);
    state->getStatusCalls++;
    *status = state->getStatusCalls >= g_fakePluginScenario.readyAfterCalls ? 0 : 1;
    return HCCL_SUCCESS;
}

int32_t FakePluginGetUserRemoteMem(void *ctx, CommMem **mem, char ***tags, uint32_t *num)
{
    (void)ctx;
    g_fakePluginScenario.getUserRemoteMemCalls++;
    *mem = g_fakeRemoteMems;
    *tags = g_fakeRemoteTags;
    *num = 2;
    return HCCL_SUCCESS;
}

void FakePluginDestroyChannel(void *ctx)
{
    g_fakePluginScenario.destroyChannelCalls++;
    delete static_cast<FakePluginChannelState *>(ctx);
}

EndpointHandle MakeFakePluginEndpointHandle()
{
    static int endpointCtx = 0;
    static PluginEndpointCtx pluginEndpoint = {
        &g_fakeEndpointOps,
        &endpointCtx,
        &g_fakePluginEntry
    };
    return MAKE_PLUGIN_EP_HANDLE(&pluginEndpoint);
}

void ResetFakePluginScenario()
{
    g_fakePluginScenario = {};
    g_fakePluginScenario.readyAfterCalls = 1;
    g_fakePluginScenario.getStatusRet = HCCL_SUCCESS;
}
} // namespace

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
        const char *dfsConfig = std::getenv("HCCL_DFS_CONFIG");
        hasOldDfsConfig_ = (dfsConfig != nullptr);
        oldDfsConfig_ = hasOldDfsConfig_ ? dfsConfig : "";
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        std::cout << "A Test case in HcommCAdptTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        if (hasOldDfsConfig_) {
            setenv("HCCL_DFS_CONFIG", oldDfsConfig_.c_str(), 1);
        } else {
            unsetenv("HCCL_DFS_CONFIG");
        }
        std::cout << "A Test case in HcommCAdptTest TearDown" << std::endl;
    }

    bool hasOldDfsConfig_ = false;
    std::string oldDfsConfig_;
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

TEST_F(HcommCAdptTest, ut_HcommChannelGetRemoteMems_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char* memInfosStorage[2] = {nullptr, nullptr};
    char** memInfos = memInfosStorage;
    MOCKER(ChannelProcess::ChannelGetRemoteMems)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGetRemoteMems(channelHandle, &memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetRemoteMems_Plugin_When_Normal_Expect_DispatchToPluginOps)
{
    ResetFakePluginScenario();
    FakePluginChannelState channelState;
    PluginChannelCtx pluginChannel = {
        &g_fakeChannelOps,
        &channelState,
        &g_fakePluginEntry
    };
    ChannelHandle channelHandle = MAKE_PLUGIN_CH_HANDLE(&pluginChannel);
    CommMem *remoteMem = nullptr;
    uint32_t memNum = 0;
    char **memInfos = nullptr;

    HcommResult ret = HcommChannelGetRemoteMems(channelHandle, &memNum, &remoteMem, &memInfos);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_fakePluginScenario.getUserRemoteMemCalls, 1U);
    EXPECT_EQ(memNum, 2U);
    EXPECT_EQ(remoteMem, g_fakeRemoteMems);
    EXPECT_EQ(memInfos, g_fakeRemoteTags);
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

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_Plugin_When_StatusReadyLater_Expect_BlockUntilReady)
{
    ResetFakePluginScenario();
    g_fakePluginScenario.readyAfterCalls = 3;
    EndpointHandle endpointHandle = MakeFakePluginEndpointHandle();
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};

    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_fakePluginScenario.createChannelCalls, 1);
    EXPECT_TRUE(IS_PLUGIN_HANDLE(channels[0]));
    auto *pluginChannel = PLUGIN_CH_CTX(channels[0]);
    auto *state = static_cast<FakePluginChannelState *>(pluginChannel->ctx);
    EXPECT_GE(state->getStatusCalls, 3U);
    EXPECT_EQ(g_fakePluginScenario.destroyChannelCalls, 0);
    EXPECT_EQ(HcommChannelDestroy(channels, 1), HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_Plugin_When_GetStatusError_Expect_DestroyAndReturnError)
{
    ResetFakePluginScenario();
    g_fakePluginScenario.getStatusRet = HCCL_E_PARA;
    EndpointHandle endpointHandle = MakeFakePluginEndpointHandle();
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};

    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);

    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(g_fakePluginScenario.createChannelCalls, 1);
    EXPECT_EQ(g_fakePluginScenario.destroyChannelCalls, 1);
    EXPECT_EQ(channels[0], 0);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_Plugin_When_Timeout_Expect_DestroyAndReturnTimeout)
{
    ResetFakePluginScenario();
    g_fakePluginScenario.readyAfterCalls = 0xFFFFFFFF;
    EndpointHandle endpointHandle = MakeFakePluginEndpointHandle();
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut)
        .stubs()
        .with(any())
        .will(returnValue((s32)(0)));

    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);

    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    EXPECT_EQ(g_fakePluginScenario.createChannelCalls, 1);
    EXPECT_EQ(g_fakePluginScenario.destroyChannelCalls, 1);
    EXPECT_EQ(channels[0], 0);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_Plugin_When_EngineNotCpu_Expect_NotSupport)
{
    ResetFakePluginScenario();
    EndpointHandle endpointHandle = MakeFakePluginEndpointHandle();
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};

    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU, &channelDesc, 1, channels);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(g_fakePluginScenario.createChannelCalls, 0);
    EXPECT_EQ(g_fakePluginScenario.destroyChannelCalls, 0);
}

TEST_F(HcommCAdptTest, ut_NicPluginDispatcher_When_NonPluginHandle_Expect_NotHandled)
{
    bool handled = true;
    void *endpoint = reinterpret_cast<void *>(0x1);
    EXPECT_EQ(PluginEndpointGet(reinterpret_cast<EndpointHandle>(0x1234), &endpoint, handled), HCCL_SUCCESS);
    EXPECT_FALSE(handled);
    EXPECT_EQ(endpoint, reinterpret_cast<void *>(0x1));

    int32_t status = 7;
    EXPECT_EQ(PluginChannelGetStatus(0x1234, &status, handled), HCCL_SUCCESS);
    EXPECT_FALSE(handled);
    EXPECT_EQ(status, 7);
}

TEST_F(HcommCAdptTest, ut_NicPluginDispatcherChannelCreate_When_NonPluginEndpoint_Expect_NotHandled)
{
    bool handled = true;
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0x1234};

    EXPECT_EQ(PluginChannelCreate(reinterpret_cast<EndpointHandle>(0x1234), COMM_ENGINE_CPU,
        &channelDesc, 1, channels, handled), HCCL_SUCCESS);
    EXPECT_FALSE(handled);
    EXPECT_EQ(channels[0], 0x1234);
}

TEST_F(HcommCAdptTest, ut_NicPluginDispatcher_When_OpUnsupported_Expect_NotSupport)
{
    static int endpointCtx = 0;
    bool handled = false;
    HcommNicChannelOps unsupportedChannelOps = {
        {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, sizeof(HcommNicChannelOps), 0}
    };
    PluginChannelCtx pluginChannel = {
        &unsupportedChannelOps,
        &endpointCtx,
        &g_fakePluginEntry
    };
    ChannelHandle channelHandle = MAKE_PLUGIN_CH_HANDLE(&pluginChannel);
    int32_t status = 0;
    EXPECT_EQ(PluginChannelGetStatus(channelHandle, &status, handled), HCCL_E_NOT_SUPPORT);
    EXPECT_TRUE(handled);

    uint32_t memNum = 0;
    CommMem *remoteMem = nullptr;
    char **memInfos = nullptr;
    EXPECT_EQ(PluginChannelGetRemoteMems(channelHandle, &memNum, &remoteMem, &memInfos, handled), HCCL_E_NOT_SUPPORT);
    EXPECT_TRUE(handled);
}

TEST_F(HcommCAdptTest, ut_NicPluginValidateInfo_When_HeaderInvalid_Expect_Fail)
{
    HcommNicPluginInfo info = {
        {HCOMM_NIC_PLUGIN_INFO_VERSION, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD, sizeof(HcommNicPluginInfo), 0},
        "fake_plugin",
        1,
        {COMM_PROTOCOL_ROCE}
    };

    EXPECT_TRUE(ValidatePluginInfo("fake_plugin.so", &info, FakePluginCreateEndpoint, FakePluginCreateChannel));

    info.header.magicWord = 0U;
    EXPECT_FALSE(ValidatePluginInfo("fake_plugin.so", &info, FakePluginCreateEndpoint, FakePluginCreateChannel));

    info.header.magicWord = HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD;
    info.header.version = HCOMM_NIC_PLUGIN_INFO_VERSION + 1U;
    EXPECT_FALSE(ValidatePluginInfo("fake_plugin.so", &info, FakePluginCreateEndpoint, FakePluginCreateChannel));

    info.header.version = HCOMM_NIC_PLUGIN_INFO_VERSION;
    info.header.size = offsetof(HcommNicPluginInfo, protocols);
    EXPECT_FALSE(ValidatePluginInfo("fake_plugin.so", &info, FakePluginCreateEndpoint, FakePluginCreateChannel));
}

TEST_F(HcommCAdptTest, ut_HcommEndpointCreate_Plugin_When_HostProtocolRegistered_Expect_PluginHandle)
{
    ResetFakePluginScenario();
    const NicPluginEntry *fakePluginEntry = &g_fakePluginEntry;
    MOCKER(hcomm::FindHostNicPlugin).stubs().will(returnValue(fakePluginEntry));
    EndpointDesc endpointDesc{};
    ASSERT_EQ(EndpointDescInit(&endpointDesc, 1), HCCL_SUCCESS);
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    EndpointHandle endpointHandle = nullptr;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_TRUE(IS_PLUGIN_HANDLE(endpointHandle));
    EXPECT_EQ(g_fakePluginScenario.createEndpointCalls, 1U);
    EXPECT_EQ(g_fakePluginScenario.initEndpointCalls, 1U);

    void *endpoint = nullptr;
    EXPECT_EQ(HcommEndpointGet(endpointHandle, &endpoint), HCCL_SUCCESS);
    ASSERT_NE(endpoint, nullptr);
    EXPECT_TRUE(static_cast<FakePluginEndpointState *>(endpoint)->inited);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = endpoint;
    mem.size = sizeof(FakePluginEndpointState);
    HcommMemHandle memHandle = nullptr;
    EXPECT_EQ(HcommMemReg(endpointHandle, "fake_mem", &mem, &memHandle), HCCL_SUCCESS);
    EXPECT_EQ(memHandle, reinterpret_cast<HcommMemHandle>(FAKE_PLUGIN_MEM_HANDLE));
    EXPECT_EQ(g_fakePluginScenario.registerMemoryCalls, 1U);

    EXPECT_EQ(HcommMemUnreg(endpointHandle, memHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakePluginScenario.unregisterMemoryCalls, 1U);
    EXPECT_EQ(g_fakePluginScenario.lastMemHandle, memHandle);

    EXPECT_EQ(HcommEndpointDestroy(endpointHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakePluginScenario.destroyEndpointCalls, 1U);
}

TEST_F(HcommCAdptTest, ut_HcommEndpoint_Plugin_When_OpUnsupported_Expect_NotSupport)
{
    static int endpointCtx = 0;
    PluginEndpointCtx pluginEndpoint = {
        &g_fakeUnsupportedEndpointOps,
        &endpointCtx,
        &g_fakePluginEntry
    };
    EndpointHandle endpointHandle = MAKE_PLUGIN_EP_HANDLE(&pluginEndpoint);
    uint32_t port = 0;

    void *memDesc = nullptr;
    uint32_t memDescLen = 0;
    char desc[8] = {};
    CommMem outMem{};
    HcommMemGrantInfo grantInfo{};
    EXPECT_EQ(HcommEndpointStartListen(endpointHandle, 60001, nullptr), HCCL_E_NOT_FOUND);
    EXPECT_EQ(HcommEndpointStopListen(endpointHandle, 60001), HCCL_E_NOT_FOUND);
    EXPECT_EQ(HcommMemExport(endpointHandle, reinterpret_cast<HcommMemHandle>(&endpointCtx), &memDesc, &memDescLen),
        HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommMemImport(endpointHandle, desc, sizeof(desc), &outMem), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommMemUnimport(endpointHandle, desc, sizeof(desc)), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommMemGrant(endpointHandle, &grantInfo), HCCL_E_NOT_FOUND);
    EXPECT_EQ(HcommMemGetAllMemHandles(endpointHandle, reinterpret_cast<void **>(&endpointCtx), &port),
        HCCL_E_NOT_FOUND);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_Plugin_When_OpsProvided_Expect_NotSupport)
{
    static int endpointCtx = 0;
    PluginEndpointCtx pluginEndpoint = {
        &g_fakeEndpointOps,
        &endpointCtx,
        &g_fakePluginEntry
    };
    EndpointHandle endpointHandle = MAKE_PLUGIN_EP_HANDLE(&pluginEndpoint);
    uint32_t port = 0;

    EXPECT_EQ(HcommEndpointGetListenPort(endpointHandle, &port), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommCAdptTest, ut_HcommEndpoint_PluginMemOps_When_Normal_Expect_DispatchToPluginOps)
{
    ResetFakePluginScenario();
    EndpointHandle endpointHandle = MakeFakePluginEndpointHandle();
    HcommMemHandle memHandle = reinterpret_cast<HcommMemHandle>(FAKE_PLUGIN_MEM_HANDLE);
    void *memDesc = nullptr;
    uint32_t memDescLen = 0;

    EXPECT_EQ(HcommMemExport(endpointHandle, memHandle, &memDesc, &memDescLen), HCCL_SUCCESS);
    EXPECT_EQ(g_fakePluginScenario.memoryExportCalls, 1U);
    EXPECT_EQ(g_fakePluginScenario.lastMemHandle, memHandle);
    EXPECT_EQ(memDesc, g_fakePluginMemDesc);
    EXPECT_EQ(memDescLen, sizeof(g_fakePluginMemDesc));

    CommMem importedMem{};
    EXPECT_EQ(HcommMemImport(endpointHandle, memDesc, memDescLen, &importedMem), HCCL_SUCCESS);
    EXPECT_EQ(g_fakePluginScenario.memoryImportCalls, 1U);
    EXPECT_EQ(g_fakePluginScenario.lastMemDesc, memDesc);
    EXPECT_EQ(g_fakePluginScenario.lastMemDescLen, memDescLen);
    EXPECT_EQ(importedMem.addr, memDesc);
    EXPECT_EQ(importedMem.size, memDescLen);

    EXPECT_EQ(HcommMemUnimport(endpointHandle, memDesc, memDescLen), HCCL_SUCCESS);
    EXPECT_EQ(g_fakePluginScenario.memoryUnimportCalls, 1U);
    EXPECT_EQ(g_fakePluginScenario.lastMemDesc, memDesc);
    EXPECT_EQ(g_fakePluginScenario.lastMemDescLen, memDescLen);

    HcommMemGrantInfo grantInfo{7, 9};
    EXPECT_EQ(HcommMemGrant(endpointHandle, &grantInfo), HCCL_E_NOT_FOUND);

    void *memHandles[2] = {};
    uint32_t memHandleNum = 0;
    EXPECT_EQ(HcommMemGetAllMemHandles(endpointHandle, memHandles, &memHandleNum), HCCL_E_NOT_FOUND);
    EXPECT_EQ(memHandleNum, 0U);
    EXPECT_EQ(memHandles[0], nullptr);
    EXPECT_EQ(memHandles[1], nullptr);
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

TEST_F(HcommCAdptTest, ut_HcommChannelDescInit_When_Normal_Expect_Version2AndFullSize)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    EXPECT_EQ(channelDesc.header.size, sizeof(HcommChannelDesc));
    EXPECT_GE(sizeof(HcommChannelDesc), HCOMM_CHANNEL_DESC_ABI_V1_SIZE + sizeof(uint32_t));
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

namespace {

uint32_t gCapturedChannelDescQos = 0U;
const char *gCapturedChannelDescChannelName = nullptr;

HcclResult CaptureCreateChannelsLoop(EndpointHandle, CommEngine, HcommChannelDesc *channelDescs, uint32_t channelNum,
    ChannelHandle *)
{
    if (channelDescs != nullptr && channelNum > 0U) {
        gCapturedChannelDescQos = channelDescs[0].qos;
        gCapturedChannelDescChannelName = channelDescs[0].channelName;
    }
    return HCCL_SUCCESS;
}

} // namespace

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_V1Desc_ClearsQosField)
{
    gCapturedChannelDescQos = 0U;
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.header.version = HCOMM_CHANNEL_VERSION_ONE;
    channelDesc.header.size = HCOMM_CHANNEL_DESC_ABI_V1_SIZE;
    channelDesc.qos = 5U;
    ChannelHandle channels[1] = {0};

    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));

    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(gCapturedChannelDescQos, 0xFFFFFFFFU);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_CurrentDescWithSmallSize_KeepsFieldsByVersion)
{
    static const char channelName[] = "channel-name";
    gCapturedChannelDescQos = 0U;
    gCapturedChannelDescChannelName = nullptr;
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.header.version = HCOMM_CHANNEL_VERSION;
    channelDesc.header.size = HCOMM_CHANNEL_DESC_ABI_V1_SIZE;
    channelDesc.qos = 5U;
    channelDesc.channelName = channelName;
    ChannelHandle channels[1] = {0};

    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));

    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(gCapturedChannelDescQos, 5U);
    EXPECT_STREQ(gCapturedChannelDescChannelName, channelName);
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
    explicit StubEndpoint() : Endpoint(EndpointDesc{}) {}
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult ServerSocketListen(const uint32_t port) override { return HCCL_SUCCESS; }
    HcclResult RegisterMemory(HcommMem mem, const char *memTag, void **memHandle) override { return HCCL_SUCCESS; }
    HcclResult UnregisterMemory(void* memHandle) override { return HCCL_SUCCESS; }
    HcclResult MemoryExport(void *memHandle, void **memDesc, uint32_t *memDescLen) override { return HCCL_SUCCESS; }
    HcclResult MemoryImport(const void *memDesc, uint32_t descLen, HcommMem *outMem) override { return HCCL_SUCCESS; }
    HcclResult MemoryUnimport(const void *memDesc, uint32_t descLen) override { return HCCL_SUCCESS; }
    HcclResult GetAllMemHandles(void **memHandles, uint32_t *memHandleNum) override { return HCCL_SUCCESS; }
};

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_ServerSocketNotSupport_Expect_E_NOT_SUPPORT)
{
    uint32_t port = 0;
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    StubEndpoint stubEndpoint;

    MOCKER_CPP(&HcommEndpointMap::GetEndpoint, Endpoint*(HcommEndpointMap::*)(EndpointHandle))
        .stubs()
        .will(returnValue(static_cast<Endpoint*>(&stubEndpoint)));

    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, &port);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointCheckFeature_When_SupportedFeature_Expect_True)
{
    EndpointDesc endpointDesc{};
    (void)memset_s(&endpointDesc, sizeof(endpointDesc), 0, sizeof(endpointDesc));
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;

    MOCKER(hcomm::CommAddrToIpAddress).stubs().will(returnValue(HCCL_SUCCESS));

    s32 devId = 0;
    MOCKER(hrtGetDevice).stubs().with(outBoundP(&devId)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Hccl::HccpPeerManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    u32 devPhyId = 0;
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), outBound(devPhyId)).will(returnValue(HCCL_SUCCESS));

    void *fakeRdmaHandle = reinterpret_cast<void *>(0x12345678);
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(fakeRdmaHandle));

    s32 directFlag = DIRECT_FLAG_PCIE;
    MOCKER(RaNdaGetDirectFlag).stubs().with(mockcpp::any(), outBoundP(&directFlag)).will(returnValue(0));

    bool value = false;
    HcommResult ret = HcommEndpointCheckFeature(HCOMM_ENDPOINT_FEATURE_NDA, &endpointDesc, &value);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(value, true);
}
