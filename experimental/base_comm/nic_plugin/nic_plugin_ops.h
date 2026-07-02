/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXPERIMENTAL_PLUGIN_OPS_H
#define HCOMM_EXPERIMENTAL_PLUGIN_OPS_H

#include "hcomm_nic_plugin.h"

#include <memory>
#include <new>
#include "channel.h"
#include "hccl_mem_defs.h"
#include "log.h"
#include "param_check_pub.h"

namespace hcomm_experimental {

template <typename Traits>
class NicPluginOps {
public:
    using EndpointT = typename Traits::EndpointT;
    using ChannelT = typename Traits::ChannelT;

    static int32_t InitEndpoint(void *endpointCtx)
    {
        CHK_PTR_NULL(endpointCtx);
        return static_cast<EndpointT *>(endpointCtx)->Init();
    }

    static HcommResult CreateEndpoint(const EndpointDesc *endpoint, void **endpointCtx)
    {
        CHK_PTR_NULL(endpoint);
        CHK_PTR_NULL(endpointCtx);
        auto ep = std::unique_ptr<EndpointT>(new (std::nothrow) EndpointT(*endpoint));
        CHK_SMART_PTR_NULL(ep);
        *endpointCtx = ep.release();
        return HCCL_SUCCESS;
    }

    static void DestroyEndpoint(void *endpointCtx)
    {
        delete static_cast<EndpointT *>(endpointCtx);
    }

    static HcommResult MemReg(void *endpointCtx, const CommMem *mem, const char *memTag, void **memHandle)
    {
        CHK_PTR_NULL(endpointCtx);
        CHK_PTR_NULL(mem);
        CHK_PTR_NULL(memHandle);
        return static_cast<EndpointT *>(endpointCtx)->RegisterMemory(*mem, memTag, memHandle);
    }

    static HcommResult MemUnreg(void *endpointCtx, void *memHandle)
    {
        CHK_PTR_NULL(endpointCtx);
        return static_cast<EndpointT *>(endpointCtx)->UnregisterMemory(memHandle);
    }

    static HcommResult MemExport(void *endpointCtx, void *memHandle, void **memDesc, uint32_t *memDescLen)
    {
        CHK_PTR_NULL(endpointCtx);
        return static_cast<EndpointT *>(endpointCtx)->MemoryExport(memHandle, memDesc, memDescLen);
    }

    static HcommResult MemImport(void *endpointCtx, const void *memDesc, uint32_t descLen, CommMem *outMem)
    {
        CHK_PTR_NULL(endpointCtx);
        return static_cast<EndpointT *>(endpointCtx)->MemoryImport(memDesc, descLen, outMem);
    }

    static HcommResult MemUnimport(void *endpointCtx, const void *memDesc, uint32_t descLen)
    {
        CHK_PTR_NULL(endpointCtx);
        return static_cast<EndpointT *>(endpointCtx)->MemoryUnimport(memDesc, descLen);
    }

    static int32_t InitChannel(void *channelCtx)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->Init();
    }

    static HcommResult CreateChannel(void *endpointCtx, const HcommChannelDesc *channelDesc, void **channelCtx)
    {
        CHK_PTR_NULL(endpointCtx);
        CHK_PTR_NULL(channelDesc);
        CHK_PTR_NULL(channelCtx);
        auto ch = std::unique_ptr<ChannelT>(
            new (std::nothrow) ChannelT(reinterpret_cast<EndpointHandle>(endpointCtx), *channelDesc));
        CHK_SMART_PTR_NULL(ch);
        *channelCtx = ch.release();
        return HCCL_SUCCESS;
    }

    static void DestroyChannel(void *channelCtx)
    {
        delete static_cast<ChannelT *>(channelCtx);
    }

    static HcommResult GetStatus(void *channelCtx, int32_t *status)
    {
        CHK_PTR_NULL(channelCtx);
        CHK_PTR_NULL(status);
        auto channelStatus = static_cast<ChannelT *>(channelCtx)->GetStatus();
        *status = (channelStatus == ChannelStatus::READY) ? 0 : 1;
        if (channelStatus == ChannelStatus::FAILED || channelStatus == ChannelStatus::SOCKET_TIMEOUT) {
            return HCCL_E_NETWORK;
        }
        return HCCL_SUCCESS;
    }

    static HcommResult GetNotifyNum(void *channelCtx, uint32_t *notifyNum)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->GetNotifyNum(notifyNum);
    }

    static HcommResult GetUserRemoteMem(void *channelCtx, CommMem **remoteMem, char ***memTags, uint32_t *memNum)
    {
        return Traits::GetUserRemoteMem(static_cast<ChannelT *>(channelCtx), remoteMem, memTags, memNum);
    }

    static HcommResult UpdateMemInfo(void *channelCtx, HcommMemHandle *memHandles, uint32_t memHandleNum)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->UpdateMemInfo(memHandles, memHandleNum);
    }

    static HcommResult NotifyRecord(void *channelCtx, uint32_t remoteNotifyIdx)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->NotifyRecord(remoteNotifyIdx);
    }

    static HcommResult NotifyWait(void *channelCtx, uint32_t localNotifyIdx, uint32_t timeout)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->NotifyWait(localNotifyIdx, timeout);
    }

    static HcommResult WriteWithNotify(void *channelCtx, void *dst, const void *src, uint64_t len,
        uint32_t remoteNotifyIdx)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->WriteWithNotify(dst, src, len, remoteNotifyIdx);
    }

    static HcommResult Write(void *channelCtx, void *dst, const void *src, uint64_t len)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->Write(dst, src, len);
    }

    static HcommResult Read(void *channelCtx, void *dst, const void *src, uint64_t len)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->Read(dst, src, len);
    }

    static HcommResult Fence(void *channelCtx)
    {
        CHK_PTR_NULL(channelCtx);
        return static_cast<ChannelT *>(channelCtx)->ChannelFence();
    }

    static HcommNicEndpointOps EndpointOps()
    {
        return {
            {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
            InitEndpoint,
            MemReg,
            MemUnreg,
            MemExport,
            MemImport,
            MemUnimport,
            DestroyEndpoint
        };
    }

    static HcommNicChannelOps ChannelOps()
    {
        return {
            {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, sizeof(HcommNicChannelOps), 0},
            InitChannel,
            GetNotifyNum,
            Write,
            WriteWithNotify,
            Read,
            NotifyRecord,
            NotifyWait,
            Fence,
            DestroyChannel,
            GetStatus,
            UpdateMemInfo,
            GetUserRemoteMem
        };
    }

    static int32_t CreateEndpointExport(const EndpointDesc *endpointDesc, void **outCtx,
        HcommNicEndpointOps **outOps, HcommNicEndpointOps *endpointOps)
    {
        CHK_PTR_NULL(endpointDesc);
        CHK_PTR_NULL(outCtx);
        CHK_PTR_NULL(outOps);
        CHK_RET(static_cast<HcclResult>(CreateEndpoint(endpointDesc, outCtx)));
        *outOps = endpointOps;
        return HCCL_SUCCESS;
    }

    static int32_t CreateChannelExport(void *epCtx, const HcommChannelDesc *channelDesc, void **outCtx,
        HcommNicChannelOps **outOps, HcommNicChannelOps *channelOps)
    {
        CHK_PTR_NULL(channelDesc);
        CHK_PTR_NULL(outCtx);
        CHK_PTR_NULL(outOps);
        CHK_RET(static_cast<HcclResult>(CreateChannel(epCtx, channelDesc, outCtx)));
        *outOps = channelOps;
        return HCCL_SUCCESS;
    }
};

} // namespace hcomm_experimental

#define HCOMM_EXPERIMENTAL_NIC_PLUGIN_EXPORTS(PluginOpsType)                                                      \
extern "C" const HcommNicPluginInfo *HcommNicPluginGetInfo(void)                                          \
{                                                                                                         \
    return &kPluginInfo;                                                                                  \
}                                                                                                         \
                                                                                                          \
extern "C" int32_t HcommNicPluginCreateEndpoint(const EndpointDesc *endpointDesc,                        \
    void **outCtx, HcommNicEndpointOps **outOps)                                                          \
{                                                                                                         \
    return PluginOpsType::CreateEndpointExport(endpointDesc, outCtx, outOps, &kEndpointOps);              \
}                                                                                                         \
                                                                                                          \
extern "C" int32_t HcommNicPluginCreateChannel(void *epCtx, const HcommChannelDesc *channelDesc,          \
    void **outCtx, HcommNicChannelOps **outOps)                                                           \
{                                                                                                         \
    return PluginOpsType::CreateChannelExport(epCtx, channelDesc, outCtx, outOps, &kChannelOps);          \
}

#endif // HCOMM_EXPERIMENTAL_PLUGIN_OPS_H
