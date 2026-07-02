/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nic_plugin_dispatcher.h"

#include <chrono>
#include <cstddef>
#include <vector>

#include "env_config/env_config.h"
#include "nic_plugin_manager.h"
#include "param_check_pub.h"

namespace hcomm {
namespace {
PluginEndpointCtx *GetPluginEndpointCtx(EndpointHandle handle, bool &handled)
{
    handled = IsPluginEndpoint(handle);
    return handled ? PLUGIN_EP_CTX(handle) : nullptr;
}

PluginChannelCtx *GetPluginChannelCtx(ChannelHandle handle, bool &handled)
{
    handled = IsPluginChannel(handle);
    return handled ? PLUGIN_CH_CTX(handle) : nullptr;
}

void DestroyCreatedPluginChannels(ChannelHandle *channels, uint32_t channelNum)
{
    if (channels == nullptr) {
        return;
    }
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        if (channels[idx] == 0) {
            continue;
        }
        bool handled = false;
        (void)PluginChannelDestroy(channels[idx], handled);
        channels[idx] = 0;
    }
}

HcommResult ConnectPluginChannels(ChannelHandle *channels, uint32_t channelNum)
{
    CHK_PTR_NULL(channels);
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]",
        __func__, channelNum), HCCL_E_PARA);

    const auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    const auto startTime = std::chrono::steady_clock::now();
    std::vector<int32_t> statusVec(channelNum, 1);

    while (true) {
        for (uint32_t idx = 0; idx < channelNum; ++idx) {
            bool handled = false;
            HcommResult ret = PluginChannelGetStatus(channels[idx], &statusVec[idx], handled);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[%s] PluginChannelGetStatus failed, ret[%d].", __func__, ret);
                return ret;
            }
            CHK_PRT_RET(!handled, HCCL_ERROR("[%s] channel[%u] is not plugin channel.", __func__, idx),
                HCCL_E_PARA);
        }

        bool allReady = true;
        for (uint32_t idx = 0; idx < channelNum; ++idx) {
            if (statusVec[idx] != 0) {
                allReady = false;
                break;
            }
        }
        if (allReady) {
            HCCL_INFO("[%s] SUCCESS.", __func__);
            return HCCL_SUCCESS;
        }

        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[%s] plugin channel connect timeout.", __func__);
            return HCCL_E_TIMEOUT;
        }
    }
}

#define CHK_PLUGIN_ENDPOINT_OP(ctx, op) do { \
    CHK_PTR_NULL(ctx);                       \
    CHK_PTR_NULL((ctx)->ops);                \
    if (!IsEndpointOpAvailable((ctx)->ops, offsetof(HcommNicEndpointOps, op), sizeof((ctx)->ops->op))) { \
        return UnsupportedPluginOp(__func__);\
    }                                        \
    if ((ctx)->ops->op == nullptr) {         \
        return UnsupportedPluginOp(__func__);\
    }                                        \
} while (0)

#define CHK_PLUGIN_CHANNEL_OP(ctx, op) do {  \
    CHK_PTR_NULL(ctx);                       \
    CHK_PTR_NULL((ctx)->ops);                \
    if (!IsChannelOpAvailable((ctx)->ops, offsetof(HcommNicChannelOps, op), sizeof((ctx)->ops->op))) { \
        return UnsupportedPluginOp(__func__);\
    }                                        \
    if ((ctx)->ops->op == nullptr) {         \
        return UnsupportedPluginOp(__func__);\
    }                                        \
} while (0)

#define CHK_PLUGIN_CHANNEL_OP_NAME(ctx, op, opName) do { \
    CHK_PTR_NULL(ctx);                                    \
    CHK_PTR_NULL((ctx)->ops);                             \
    if (!IsChannelOpAvailable((ctx)->ops, offsetof(HcommNicChannelOps, op), sizeof((ctx)->ops->op))) { \
        return UnsupportedPluginOp(opName);               \
    }                                                     \
    if ((ctx)->ops->op == nullptr) {                      \
        return UnsupportedPluginOp(opName);               \
    }                                                     \
} while (0)

#define DISPATCH_PLUGIN_ENDPOINT_OP(handle, handled, op, args) do { \
    PluginEndpointCtx *ctx = GetPluginEndpointCtx((handle), (handled)); \
    if (!(handled)) {                                           \
        return HCCL_SUCCESS;                                    \
    }                                                           \
    CHK_PLUGIN_ENDPOINT_OP(ctx, op);                            \
    return ctx->ops->op args;                                   \
} while (0)

#define DISPATCH_PLUGIN_CHANNEL_OP(handle, handled, op, args) do { \
    PluginChannelCtx *ctx = GetPluginChannelCtx((handle), (handled)); \
    if (!(handled)) {                                         \
        return HCCL_SUCCESS;                                  \
    }                                                         \
    CHK_PLUGIN_CHANNEL_OP(ctx, op);                           \
    return ctx->ops->op args;                                 \
} while (0)

#define DISPATCH_PLUGIN_CHANNEL_OP_NAME(handle, handled, op, opName, args) do { \
    PluginChannelCtx *ctx = GetPluginChannelCtx((handle), (handled)); \
    if (!(handled)) {                                                \
        return HCCL_SUCCESS;                                         \
    }                                                                \
    CHK_PLUGIN_CHANNEL_OP_NAME(ctx, op, opName);                     \
    return ctx->ops->op args;                                        \
} while (0)

HcommResult UnsupportedPluginChannelOp(ChannelHandle handle, bool &handled, const char *opName)
{
    handled = IsPluginChannel(handle);
    if (!handled) {
        return HCCL_SUCCESS;
    }
    return UnsupportedPluginOp(opName);
}

HcommResult UnsupportedPluginChannelReduceOp(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, bool &handled, const char *opName)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return UnsupportedPluginChannelOp(handle, handled, opName);
}

HcommResult DispatchPluginChannelWriteNbi(ChannelHandle handle, void *dst, const void *src, uint64_t len,
    bool &handled, const char *opName)
{
    DISPATCH_PLUGIN_CHANNEL_OP_NAME(handle, handled, writeNbi, opName, (ctx->ctx, dst, src, len));
}

HcommResult DispatchPluginChannelWriteWithNotifyNbi(ChannelHandle handle, void *dst, const void *src, uint64_t len,
    uint32_t remoteNotifyIdx, bool &handled, const char *opName)
{
    DISPATCH_PLUGIN_CHANNEL_OP_NAME(handle, handled, writeWithNotifyNbi, opName,
        (ctx->ctx, dst, src, len, remoteNotifyIdx));
}

HcommResult DispatchPluginChannelReadNbi(ChannelHandle handle, void *dst, const void *src, uint64_t len,
    bool &handled, const char *opName)
{
    DISPATCH_PLUGIN_CHANNEL_OP_NAME(handle, handled, readNbi, opName, (ctx->ctx, dst, src, len));
}
} // namespace

bool IsPluginEndpoint(EndpointHandle handle)
{
    return IS_PLUGIN_HANDLE(handle);
}

bool IsPluginChannel(ChannelHandle handle)
{
    return IS_PLUGIN_HANDLE(handle);
}

HcommResult PluginEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle, bool &handled)
{
    handled = false;
    CHK_PTR_NULL(endpoint);
    CHK_PTR_NULL(endpointHandle);
    if (endpoint->loc.locType != ENDPOINT_LOC_TYPE_HOST || FindHostNicPlugin(endpoint->protocol) == nullptr) {
        return HCCL_SUCCESS;
    }
    handled = true;
    return CreatePluginEndpoint(endpoint, endpointHandle);
}

HcommResult PluginEndpointGet(EndpointHandle handle, void **endpoint, bool &handled)
{
    PluginEndpointCtx *ctx = GetPluginEndpointCtx(handle, handled);
    if (!handled) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(ctx);
    *endpoint = ctx->ctx;
    return HCCL_SUCCESS;
}

HcommResult PluginEndpointDestroy(EndpointHandle handle, bool &handled)
{
    handled = IsPluginEndpoint(handle);
    return handled ? DestroyPluginEndpoint(handle) : HCCL_SUCCESS;
}

HcommResult PluginMemReg(EndpointHandle handle, const char *memTag,
    const CommMem *mem, HcommMemHandle *memHandle, bool &handled)
{
    DISPATCH_PLUGIN_ENDPOINT_OP(handle, handled, registerMemory,
        (ctx->ctx, mem, memTag, reinterpret_cast<void **>(memHandle)));
}

HcommResult PluginMemUnreg(EndpointHandle handle, HcommMemHandle memHandle, bool &handled)
{
    DISPATCH_PLUGIN_ENDPOINT_OP(handle, handled, unregisterMemory, (ctx->ctx, memHandle));
}

HcommResult PluginMemExport(EndpointHandle handle, HcommMemHandle memHandle,
    void **memDesc, uint32_t *memDescLen, bool &handled)
{
    DISPATCH_PLUGIN_ENDPOINT_OP(handle, handled, memoryExport, (ctx->ctx, memHandle, memDesc, memDescLen));
}

HcommResult PluginMemImport(EndpointHandle handle, const void *memDesc, uint32_t descLen,
    CommMem *outMem, bool &handled)
{
    DISPATCH_PLUGIN_ENDPOINT_OP(handle, handled, memoryImport, (ctx->ctx, memDesc, descLen, outMem));
}

HcommResult PluginMemUnimport(EndpointHandle handle, const void *memDesc, uint32_t descLen, bool &handled)
{
    DISPATCH_PLUGIN_ENDPOINT_OP(handle, handled, memoryUnimport, (ctx->ctx, memDesc, descLen));
}

HcommResult PluginChannelCreate(EndpointHandle endpointHandle, const HcommChannelDesc *channelDesc,
    ChannelHandle *channelHandle)
{
    return CreatePluginChannel(endpointHandle, channelDesc, channelHandle);
}

HcommResult PluginChannelCreate(EndpointHandle endpointHandle, CommEngine engine,
    const HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels, bool &handled)
{
    handled = IsPluginEndpoint(endpointHandle);
    if (!handled) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]",
        __func__, channelNum), HCCL_E_PARA);
    CHK_PRT_RET(engine != COMM_ENGINE_CPU,
        HCCL_ERROR("[%s] nic plugin endpoint only supports COMM_ENGINE_CPU, engine[%d].", __func__, engine),
        HCCL_E_NOT_SUPPORT);

    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        HcommResult ret = PluginChannelCreate(endpointHandle, &channelDescs[idx], &channels[idx]);
        if (ret != HCCL_SUCCESS) {
            DestroyCreatedPluginChannels(channels, idx);
            return ret;
        }
    }

    HcommResult ret = ConnectPluginChannels(channels, channelNum);
    if (ret != HCCL_SUCCESS) {
        DestroyCreatedPluginChannels(channels, channelNum);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcommResult PluginChannelGet(ChannelHandle handle, void **channel, bool &handled)
{
    PluginChannelCtx *ctx = GetPluginChannelCtx(handle, handled);
    if (!handled) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(ctx);
    *channel = ctx->ctx;
    return HCCL_SUCCESS;
}

HcommResult PluginChannelGetStatus(ChannelHandle handle, int32_t *status, bool &handled)
{
    DISPATCH_PLUGIN_CHANNEL_OP(handle, handled, getStatus, (ctx->ctx, status));
}

HcommResult PluginChannelGetNotifyNum(ChannelHandle handle, uint32_t *notifyNum, bool &handled)
{
    DISPATCH_PLUGIN_CHANNEL_OP(handle, handled, getNotifyNum, (ctx->ctx, notifyNum));
}

HcommResult PluginChannelDestroy(ChannelHandle handle, bool &handled)
{
    handled = IsPluginChannel(handle);
    return handled ? DestroyPluginChannel(handle) : HCCL_SUCCESS;
}

HcommResult PluginChannelUpdateMemInfo(ChannelHandle handle, HcommMemHandle *memHandles, uint32_t memHandleNum,
    bool &handled)
{
    DISPATCH_PLUGIN_CHANNEL_OP(handle, handled, updateMemInfo, (ctx->ctx, memHandles, memHandleNum));
}

HcommResult PluginChannelGetRemoteMems(ChannelHandle handle, uint32_t *memNum, CommMem **remoteMem,
    char ***memInfos, bool &handled)
{
    DISPATCH_PLUGIN_CHANNEL_OP(handle, handled, getUserRemoteMem, (ctx->ctx, remoteMem, memInfos, memNum));
}

HcommResult PluginChannelWrite(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src, uint64_t len,
    bool &handled)
{
    (void)thread;
    return DispatchPluginChannelWriteNbi(handle, dst, src, len, handled, __func__);
}

HcommResult PluginChannelBatchTransfer(ChannelHandle handle, ThreadHandle thread,
    const HcommBatchTransferDesc *transferDescs, uint32_t transferDescNum, bool &handled)
{
    (void)thread;
    (void)transferDescs;
    (void)transferDescNum;
    return UnsupportedPluginChannelOp(handle, handled, __func__);
}

HcommResult PluginChannelWriteReduce(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, bool &handled)
{
    return UnsupportedPluginChannelReduceOp(handle, thread, dst, src, count, dataType, reduceOp, handled, __func__);
}

HcommResult PluginChannelWriteWithNotify(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx, bool &handled)
{
    (void)thread;
    return DispatchPluginChannelWriteWithNotifyNbi(handle, dst, src, len, remoteNotifyIdx, handled, __func__);
}

HcommResult PluginChannelWriteReduceWithNotify(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, uint32_t remoteNotifyIdx, bool &handled)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    (void)remoteNotifyIdx;
    return UnsupportedPluginChannelOp(handle, handled, __func__);
}

HcommResult PluginChannelRead(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src, uint64_t len,
    bool &handled)
{
    (void)thread;
    return DispatchPluginChannelReadNbi(handle, dst, src, len, handled, __func__);
}

HcommResult PluginChannelReadReduce(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, bool &handled)
{
    return UnsupportedPluginChannelReduceOp(handle, thread, dst, src, count, dataType, reduceOp, handled, __func__);
}

HcommResult PluginChannelWriteNbi(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src, uint64_t len,
    bool &handled)
{
    (void)thread;
    return DispatchPluginChannelWriteNbi(handle, dst, src, len, handled, __func__);
}

HcommResult PluginChannelWriteWithNotifyNbi(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx, bool &handled)
{
    (void)thread;
    return DispatchPluginChannelWriteWithNotifyNbi(handle, dst, src, len, remoteNotifyIdx, handled, __func__);
}

HcommResult PluginChannelReadNbi(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src, uint64_t len,
    bool &handled)
{
    (void)thread;
    return DispatchPluginChannelReadNbi(handle, dst, src, len, handled, __func__);
}

HcommResult PluginChannelNotifyRecord(ChannelHandle handle, ThreadHandle thread, uint32_t remoteNotifyIdx,
    bool &handled)
{
    (void)thread;
    DISPATCH_PLUGIN_CHANNEL_OP(handle, handled, notifyRecord, (ctx->ctx, remoteNotifyIdx));
}

HcommResult PluginChannelNotifyWait(ChannelHandle handle, ThreadHandle thread, uint32_t localNotifyIdx,
    uint32_t timeOut, bool &handled)
{
    (void)thread;
    DISPATCH_PLUGIN_CHANNEL_OP(handle, handled, notifyWait, (ctx->ctx, localNotifyIdx, timeOut));
}

HcommResult PluginChannelFence(ChannelHandle handle, ThreadHandle thread, bool &handled)
{
    (void)thread;
    DISPATCH_PLUGIN_CHANNEL_OP(handle, handled, fence, (ctx->ctx));
}

} // namespace hcomm
