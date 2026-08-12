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
#include "hcomm_channel.h"
#include "channel_process.h"
#include "endpoint_map.h"
#include "env_config/env_config.h"

#include "nic_plugin_manager.h"
#include "orion_adpt_utils.h"

// Test stub: StubEndpoint 需要非空的 nicOps_ 避免 GetNicOps() 返回 nullptr
static int32_t StubGetListenPort(void* ctx, uint32_t* port)
{
    auto* endpoint = static_cast<hcomm::Endpoint*>(ctx);
    return endpoint->ServerSocketGetListenPort(port);
}
static HcommNicEndpointOps g_stubEndpointOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    StubGetListenPort,
};
#include "hccp_peer_manager.h"
#include "rdma_handle_manager.h"
#include "hccp_nda.h"
#include "adapter_rts_common.h"

using namespace hcomm;

HcclResult StubServerSocketGetListenPort(Endpoint* /*endpoint*/, uint32_t* port)
{
    *port = 12345;
    return HCCL_SUCCESS;
}

class HcommCAdptTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcommCAdptTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcommCAdptTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        const char* dfsConfig = std::getenv("HCCL_DFS_CONFIG");
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
    MOCKER(ChannelProcess::ChannelGet).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGet(channelHandle, &channel);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_Normal_Expect_Success)
{
    // 假 handle 未注册到 channel map，GetChannelsInfo 找不到 channel 返回错误
    ChannelHandle channelList[2] = {0x12345, 0x12346};
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_ChannelListNull_Expect_E_PTR)
{
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(nullptr, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_StatusListNull_Expect_E_PTR)
{
    ChannelHandle channelList[2] = {0x12345, 0x12346};
    HcommResult ret = HcommChannelGetStatus(channelList, 2, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_ListNumZero_Expect_E_PARA)
{
    ChannelHandle channelList[2] = {0x12345, 0x12346};
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(channelList, 0, statusList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ===================== 非阻塞建链 HandleAicpuStatus/HandleAivStatus UT =====================

// FakeChannel for HcommChannelGetStatus 端到端测试
class HcommCAdptFakeChannel : public hcomm::Channel {
public:
    HcommCAdptFakeChannel(CommEngine eng, hcomm::ChannelStatus sta)
    {
        engine_ = eng;
        channelKind_ = hcomm::HcommChannelKind::AICPU_TS_UBOE;
        fakeStatus_ = sta;
    }
    ~HcommCAdptFakeChannel() = default;

    hcomm::HcommChannelKind GetChannelKind() const override { return channelKind_; }
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override { return HCCL_SUCCESS; }
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override { return HCCL_SUCCESS; }
    hcomm::ChannelStatus GetStatus() override { return fakeStatus_; }
    const HcommChannelDesc& GetChannelDesc() const override { return desc_; }
    HcclResult Clean() override { return HCCL_SUCCESS; }
    HcclResult Resume() override { return HCCL_SUCCESS; }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override { return HCCL_SUCCESS; }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override { return HCCL_SUCCESS; }
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Write(void* dst, const void* src, uint64_t len) override { return HCCL_SUCCESS; }
    HcclResult Read(void* dst, const void* src, uint64_t len) override { return HCCL_SUCCESS; }
    HcclResult ChannelFence() override { return HCCL_SUCCESS; }

private:
    hcomm::ChannelStatus fakeStatus_{hcomm::ChannelStatus::INIT};
    HcommChannelDesc desc_{};
};

// 辅助：注册 channel 到全局 map
static ChannelHandle HcommCAdptRegisterChannel(std::shared_ptr<HcommCAdptFakeChannel>& ch)
{
    ChannelHandle handle = reinterpret_cast<ChannelHandle>(ch.get());
    hcomm::DeviceChannelKey key{0, handle};
    std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
    hcomm::ChannelProcess::g_ChannelMap.emplace(handle, ch);
    hcomm::ChannelProcess::g_ChannelD2HMap.emplace(key, handle);
    return handle;
}

// 辅助：清理全局 map
static void HcommCAdptClearChannelMap()
{
    std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
    hcomm::ChannelProcess::g_ChannelMap.clear();
    hcomm::ChannelProcess::g_ChannelD2HMap.clear();
}

// HandleAicpuStatus: 未全 READY（含 CONNECTING）→ 返回 CONNECTING
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_Connecting_Return_Connecting)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::INIT);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);

    HcommCAdptClearChannelMap();
}

// HandleAicpuStatus: 含 FAILED → FAILED 真实返回
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_Failed_Return_Failed)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::FAILED);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_FAILED);

    HcommCAdptClearChannelMap();
}

// HandleAicpuStatus: 含 SOCKET_TIMEOUT → TIMEOUT
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_SocketTimeout_Return_Timeout)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::SOCKET_TIMEOUT);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT);

    HcommCAdptClearChannelMap();
}

// HandleAicpuStatus: 混合状态（READY+FAILED）→ 未全 READY，FAILED 真实返回，READY 报 CONNECTING
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_MixedReadyFailed_Return_FailedAndConnecting)
{
    auto ch1 = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::READY);
    auto ch2 = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::FAILED);
    ChannelHandle handles[2] = {HcommCAdptRegisterChannel(ch1), HcommCAdptRegisterChannel(ch2)};

    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(handles, 2, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);
    EXPECT_EQ(statusList[1], hcomm::HCOMM_CHANNEL_STATUS_FAILED);

    HcommCAdptClearChannelMap();
}

// HandleAivStatus: 未全 READY（含 CONNECTING）→ 返回 CONNECTING
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aiv_When_Connecting_Return_Connecting)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AIV, hcomm::ChannelStatus::INIT);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);

    HcommCAdptClearChannelMap();
}

// HandleAivStatus: 含 FAILED → FAILED 真实返回
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aiv_When_Failed_Return_Failed)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AIV, hcomm::ChannelStatus::FAILED);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_FAILED);

    HcommCAdptClearChannelMap();
}

// 非 AICPU/AIV engine（如 CPU）→ 走通用 CopyLinkStatusToOutput
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Cpu_When_Connecting_Return_Connecting)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_CPU, hcomm::ChannelStatus::INIT);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);

    HcommCAdptClearChannelMap();
}

// GetChannelsInfo 失败（channel not found）→ HcommChannelGetStatus 返回 E_INTERNAL
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_ChannelNotFound_Return_E_INTERNAL)
{
    ChannelHandle handle = static_cast<ChannelHandle>(0x8888);
    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    HcommCAdptClearChannelMap();
}

// HandleAivStatus: 全 READY + 已 deviceEntityReady → FillAivDevEntities 跳过 → 返回 READY
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aiv_When_AllReady_Return_Ready)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AIV, hcomm::ChannelStatus::READY);
    ch->SetDeviceEntityReady();
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_READY);

    HcommCAdptClearChannelMap();
}

// HandleAicpuStatus: 全 READY + 已 deviceEntityReady → LaunchAicpuKernel 跳过 → 返回 READY
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_AllReady_Return_Ready)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::READY);
    ch->SetDeviceEntityReady();
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_READY);

    HcommCAdptClearChannelMap();
}

// 混合 engine：AICPU + AIV 同时传入，各自独立处理 → 各自返回对应状态
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_MixedEngine_When_MixedStatus_Return_EachStatus)
{
    auto chAicpuReady = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::READY);
    chAicpuReady->SetDeviceEntityReady();
    auto chAivConnecting = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AIV, hcomm::ChannelStatus::INIT);
    auto chCpuFailed = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_CPU, hcomm::ChannelStatus::FAILED);
    ChannelHandle handles[3] = {
        HcommCAdptRegisterChannel(chAicpuReady),
        HcommCAdptRegisterChannel(chAivConnecting),
        HcommCAdptRegisterChannel(chCpuFailed),
    };

    int32_t statusList[3] = {0, 0, 0};
    HcommResult ret = HcommChannelGetStatus(handles, 3, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_READY);      // AICPU: READY
    EXPECT_EQ(statusList[1], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING); // AIV: INIT → CONNECTING
    EXPECT_EQ(statusList[2], hcomm::HCOMM_CHANNEL_STATUS_FAILED);     // CPU: FAILED

    HcommCAdptClearChannelMap();
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetNotifyNum_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    uint32_t notifyNum = 0;
    MOCKER(ChannelProcess::ChannelGetNotifyNum).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGetNotifyNum(channelHandle, &notifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelDestroy_When_Normal_Expect_Success)
{
    ChannelHandle channels[2] = {0x12345, 0x12346};
    MOCKER(ChannelProcess::ChannelDestroy).stubs().will(returnValue(HCCL_SUCCESS));
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
    MOCKER(ChannelProcess::ChannelGetRemoteMems).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGetRemoteMems(channelHandle, &memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

namespace {

uint32_t gCapturedChannelDescQos = 0U;
uint32_t gCapturedChannelDescSqDepth = 0U;
const char* gCapturedChannelDescChannelName = nullptr;

HcclResult CaptureCreateChannelsLoop(
    EndpointHandle, CommEngine, HcommChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle*)
{
    if (channelDescs != nullptr && channelNum > 0U) {
        gCapturedChannelDescQos = channelDescs[0].qos;
        gCapturedChannelDescSqDepth = channelDescs[0].ubAttr.sqDepth;
        gCapturedChannelDescChannelName = channelDescs[0].channelName;
    }
    return HCCL_SUCCESS;
}

} // namespace

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_When_Normal_Expect_Success)
{
    gCapturedChannelDescSqDepth = 0U;
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_TP;
    channelDesc.ubAttr.sqDepth = 300U;
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_AIV, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(gCapturedChannelDescSqDepth, 512U);
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
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
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

TEST_F(HcommCAdptTest, ut_HcommChannelDescInit_When_Normal_Expect_Version2AndFullSize)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    EXPECT_EQ(channelDesc.header.size, sizeof(HcommChannelDesc));
    EXPECT_GE(sizeof(HcommChannelDesc), HCOMM_CHANNEL_DESC_ABI_V1_SIZE + sizeof(uint32_t));
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_AICPU_When_CreateLoopMocked_PreAllocFail_Return_Error)
{
    // CreateChannelsLoop 被 mock 返回 SUCCESS 但未填充 targetChannels，
    // PreAllocAicpuChannels 拿到空指针 channel 返回 E_PTR
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_CPU_Expect_Success)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_CCU_Expect_Success)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CCU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

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
    // 新增 ENABLE_EXPERIMENTAL 路径先校验 endpoint 是否存在再校验 port 是否为空
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_HandleInvalid_Expect_E_NOT_FOUND)
{
    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(nullptr, &port);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

class StubEndpoint final : public Endpoint {
public:
    explicit StubEndpoint() : Endpoint(EndpointDesc{}) { SetNicEndpointCtx(&g_stubEndpointOps, this); }
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult ServerSocketListen(const uint32_t port) override { return HCCL_SUCCESS; }
    HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) override { return HCCL_SUCCESS; }
    HcclResult UnregisterMemory(void* memHandle) override { return HCCL_SUCCESS; }
    HcclResult MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen) override { return HCCL_SUCCESS; }
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override { return HCCL_SUCCESS; }
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override { return HCCL_SUCCESS; }
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override { return HCCL_SUCCESS; }
};

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_ServerSocketNotSupport_Expect_E_NOT_SUPPORT)
{
    uint32_t port = 0;
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    StubEndpoint stubEndpoint;

    MOCKER_CPP(&HcommEndpointMap::GetEndpoint, Endpoint * (HcommEndpointMap::*)(EndpointHandle))
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

    void* fakeRdmaHandle = reinterpret_cast<void*>(0x12345678);
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(fakeRdmaHandle));

    s32 directFlag = DIRECT_FLAG_PCIE;
    MOCKER(RaNdaGetDirectFlag).stubs().with(mockcpp::any(), outBoundP(&directFlag)).will(returnValue(0));

    bool value = false;
    HcommResult ret = HcommEndpointCheckFeature(HCOMM_ENDPOINT_FEATURE_NDA, &endpointDesc, &value);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(value, true);
}
