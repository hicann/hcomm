/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_NIC_PLUGIN_H
#define HCOMM_NIC_PLUGIN_H

#include <stdint.h>

#include "hcomm_primitives.h"
#include "hcomm_res.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HCOMM_NIC_PLUGIN_MAX_PROTOCOLS 4U
#define HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD 0x0fcf0f1fU
#define HCOMM_NIC_PLUGIN_INFO_VERSION 1U
#define HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD 0x0fcf0f2fU
#define HCOMM_NIC_ENDPOINT_OPS_VERSION 1U
#define HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD 0x0fcf0f3fU
#define HCOMM_NIC_CHANNEL_OPS_VERSION 1U

typedef struct {
    CommAbiHeader header;
    int32_t (*init)(void *ctx);
    int32_t (*registerMemory)(void *ctx, const CommMem *mem, const char *tag, void **handle);
    int32_t (*unregisterMemory)(void *ctx, void *handle);
    int32_t (*memoryExport)(void *ctx, void *handle, void **desc, uint32_t *descLen);
    int32_t (*memoryImport)(void *ctx, const void *desc, uint32_t descLen, CommMem *outMem);
    int32_t (*memoryUnimport)(void *ctx, const void *desc, uint32_t descLen);
    void (*destroy)(void *ctx);
} HcommNicEndpointOps;

typedef struct {
    CommAbiHeader header;
    int32_t (*init)(void *ctx);
    int32_t (*getNotifyNum)(void *ctx, uint32_t *num);
    int32_t (*writeNbi)(void *ctx, void *dst, const void *src, uint64_t len);
    int32_t (*writeWithNotifyNbi)(void *ctx, void *dst, const void *src, uint64_t len, uint32_t remoteNotifyIdx);
    int32_t (*readNbi)(void *ctx, void *dst, const void *src, uint64_t len);
    int32_t (*notifyRecord)(void *ctx, uint32_t remoteNotifyIdx);
    int32_t (*notifyWait)(void *ctx, uint32_t localNotifyIdx, uint32_t timeout);
    int32_t (*fence)(void *ctx);
    void (*destroy)(void *ctx);
    int32_t (*getStatus)(void *ctx, int32_t *status);
    int32_t (*updateMemInfo)(void *ctx, HcommMemHandle *memHandles, uint32_t memHandleNum);
    int32_t (*getUserRemoteMem)(void *ctx, CommMem **mem, char ***tags, uint32_t *num);
} HcommNicChannelOps;

typedef struct {
    CommAbiHeader header;
    const char *name;
    uint32_t protocolCount;
    CommProtocol protocols[HCOMM_NIC_PLUGIN_MAX_PROTOCOLS];
    uint64_t reserved[8];
} HcommNicPluginInfo;

typedef const HcommNicPluginInfo *(*HcommNicPluginGetInfoFunc)(void);
typedef int32_t (*HcommNicPluginCreateEndpointFunc)(
    const EndpointDesc *endpointDesc, void **outCtx, HcommNicEndpointOps **outOps);
typedef int32_t (*HcommNicPluginCreateChannelFunc)(
    void *epCtx, const HcommChannelDesc *channelDesc, void **outCtx, HcommNicChannelOps **outOps);

const HcommNicPluginInfo *HcommNicPluginGetInfo(void);
int32_t HcommNicPluginCreateEndpoint(
    const EndpointDesc *endpointDesc, void **outCtx, HcommNicEndpointOps **outOps);
int32_t HcommNicPluginCreateChannel(
    void *epCtx, const HcommChannelDesc *channelDesc, void **outCtx, HcommNicChannelOps **outOps);

#ifdef __cplusplus
}
#endif

#endif
