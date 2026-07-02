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

#include "hccl_api_data.h"
#include "hcomm_nic_plugin.h"
#include "hcomm_primitives.h"
#include "nic_plugin_manager.h"

namespace {
struct FakePluginChannelState {
    uint32_t writeNbiCalls = 0;
    uint32_t writeWithNotifyNbiCalls = 0;
    uint32_t readNbiCalls = 0;
    uint32_t notifyRecordCalls = 0;
    uint32_t notifyWaitCalls = 0;
    uint32_t fenceCalls = 0;
};

HcommNicChannelOps g_fakeChannelOps{};
HcommNicChannelOps g_unsupportedChannelOps{};
FakePluginChannelState g_fakeState{};
hcomm::PluginChannelCtx g_fakePluginChannel{&g_fakeChannelOps, &g_fakeState, nullptr};
hcomm::PluginChannelCtx g_unsupportedPluginChannel{&g_unsupportedChannelOps, &g_fakeState, nullptr};

constexpr ThreadHandle FAKE_THREAD = 0x1357U;
constexpr uint32_t FAKE_NOTIFY_IDX = 3U;
constexpr uint32_t FAKE_TIMEOUT = 100U;

int32_t FakeWriteNbi(void *ctx, void *dst, const void *src, uint64_t len)
{
    (void)dst;
    (void)src;
    (void)len;
    static_cast<FakePluginChannelState *>(ctx)->writeNbiCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeWriteWithNotifyNbi(void *ctx, void *dst, const void *src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    static_cast<FakePluginChannelState *>(ctx)->writeWithNotifyNbiCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeReadNbi(void *ctx, void *dst, const void *src, uint64_t len)
{
    (void)dst;
    (void)src;
    (void)len;
    static_cast<FakePluginChannelState *>(ctx)->readNbiCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeNotifyRecord(void *ctx, uint32_t remoteNotifyIdx)
{
    (void)remoteNotifyIdx;
    static_cast<FakePluginChannelState *>(ctx)->notifyRecordCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeNotifyWait(void *ctx, uint32_t localNotifyIdx, uint32_t timeout)
{
    (void)localNotifyIdx;
    (void)timeout;
    static_cast<FakePluginChannelState *>(ctx)->notifyWaitCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeFence(void *ctx)
{
    static_cast<FakePluginChannelState *>(ctx)->fenceCalls++;
    return HCCL_SUCCESS;
}

ChannelHandle MakeFakePluginChannelHandle()
{
    return MAKE_PLUGIN_CH_HANDLE(&g_fakePluginChannel);
}

ChannelHandle MakeUnsupportedPluginChannelHandle()
{
    return MAKE_PLUGIN_CH_HANDLE(&g_unsupportedPluginChannel);
}

void InitFakeChannelOps()
{
    g_fakeChannelOps = {};
    g_fakeChannelOps.header = {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD,
        sizeof(HcommNicChannelOps), 0};
    g_fakeChannelOps.writeNbi = FakeWriteNbi;
    g_fakeChannelOps.writeWithNotifyNbi = FakeWriteWithNotifyNbi;
    g_fakeChannelOps.readNbi = FakeReadNbi;
    g_fakeChannelOps.notifyRecord = FakeNotifyRecord;
    g_fakeChannelOps.notifyWait = FakeNotifyWait;
    g_fakeChannelOps.fence = FakeFence;
}
} // namespace

class UtCpuHcommPluginChannelOps : public testing::Test {
protected:
    void SetUp() override
    {
        g_fakeState = {};
        InitFakeChannelOps();
        g_unsupportedChannelOps = {};
        g_unsupportedChannelOps.header = {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD,
            sizeof(HcommNicChannelOps), 0};
    }

    char srcBuf[8]{};
    char dstBuf[8]{};
    void *dst = dstBuf;
    const void *src = srcBuf;
    uint64_t len = sizeof(srcBuf);
    uint64_t count = 1;
    HcommBatchTransferDesc desc{};
};

TEST_F(UtCpuHcommPluginChannelOps, Ut_PluginChannel_When_WriteOpsCalled_Expect_DispatchToWriteNbi)
{
    ChannelHandle channel = MakeFakePluginChannelHandle();

    EXPECT_EQ(HcommWriteOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_SUCCESS);
    EXPECT_EQ(HcommWriteNbiOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_SUCCESS);
    EXPECT_EQ(HcommWriteNbi(channel, dst, src, len), HCCL_SUCCESS);

    EXPECT_EQ(g_fakeState.writeNbiCalls, 3U);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_PluginChannel_When_ReadOpsCalled_Expect_DispatchToReadNbi)
{
    ChannelHandle channel = MakeFakePluginChannelHandle();

    EXPECT_EQ(HcommReadOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_SUCCESS);
    EXPECT_EQ(HcommReadNbiOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_SUCCESS);
    EXPECT_EQ(HcommReadNbi(channel, dst, src, len), HCCL_SUCCESS);

    EXPECT_EQ(g_fakeState.readNbiCalls, 3U);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_PluginChannel_When_WriteWithNotifyOpsCalled_Expect_DispatchToWriteWithNotifyNbi)
{
    ChannelHandle channel = MakeFakePluginChannelHandle();

    EXPECT_EQ(HcommWriteWithNotifyOnThread(FAKE_THREAD, channel, dst, src, len, FAKE_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(HcommWriteWithNotifyNbiOnThread(FAKE_THREAD, channel, dst, src, len, FAKE_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(HcommWriteWithNotifyNbi(channel, dst, src, len, FAKE_NOTIFY_IDX), HCCL_SUCCESS);

    EXPECT_EQ(g_fakeState.writeWithNotifyNbiCalls, 3U);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_PluginChannel_When_NotifyAndFenceOpsCalled_Expect_DispatchToSameOps)
{
    ChannelHandle channel = MakeFakePluginChannelHandle();

    EXPECT_EQ(HcommChannelNotifyRecordOnThread(FAKE_THREAD, channel, FAKE_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(HcommChannelNotifyRecord(channel, FAKE_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(HcommChannelNotifyWaitOnThread(FAKE_THREAD, channel, FAKE_NOTIFY_IDX, FAKE_TIMEOUT), HCCL_SUCCESS);
    EXPECT_EQ(HcommChannelNotifyWait(channel, FAKE_NOTIFY_IDX, FAKE_TIMEOUT), HCCL_SUCCESS);
    EXPECT_EQ(CommFence(FAKE_THREAD, channel), HCCL_SUCCESS);
    EXPECT_EQ(HcommChannelFenceOnThread(FAKE_THREAD, channel), HCCL_SUCCESS);
    EXPECT_EQ(HcommChannelFence(channel), HCCL_SUCCESS);

    EXPECT_EQ(g_fakeState.notifyRecordCalls, 2U);
    EXPECT_EQ(g_fakeState.notifyWaitCalls, 2U);
    EXPECT_EQ(g_fakeState.fenceCalls, 3U);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_PluginChannel_When_ReduceOrBatchCalled_Expect_NotSupport)
{
    ChannelHandle channel = MakeFakePluginChannelHandle();
    desc.transType = HCOMM_TRANSFER_TYPE_WRITE;
    desc.transferInfo.write.dst = dst;
    desc.transferInfo.write.src = const_cast<void *>(src);
    desc.transferInfo.write.len = len;

    EXPECT_EQ(HcommWriteReduceOnThread(FAKE_THREAD, channel, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
        HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommWriteReduceWithNotifyOnThread(FAKE_THREAD, channel, dst, src, count, HCOMM_DATA_TYPE_FP32,
        HCOMM_REDUCE_SUM, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommReadReduceOnThread(FAKE_THREAD, channel, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
        HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommBatchTransferOnThread(FAKE_THREAD, channel, &desc, 1), HCCL_E_NOT_SUPPORT);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_PluginChannel_When_OpUnsupported_Expect_NotSupport)
{
    ChannelHandle channel = MakeUnsupportedPluginChannelHandle();

    EXPECT_EQ(HcommWriteOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommWriteNbi(channel, dst, src, len), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelNotifyRecord(channel, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelFence(channel), HCCL_E_NOT_SUPPORT);
}
