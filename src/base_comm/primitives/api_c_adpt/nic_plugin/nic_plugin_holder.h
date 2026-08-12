/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCOMM_NIC_PLUGIN_HOLDER_H
#define HCOMM_NIC_PLUGIN_HOLDER_H

#include "endpoint.h"
#include "channel.h"
#include "nic_plugin_manager.h"

namespace hcomm {

template <typename Ops>
void DestroyNicPluginOpsAndCtx(Ops*& nicOps, void* nicCtx)
{
    if (nicOps != nullptr) {
        if (nicOps->destroy != nullptr) {
            int32_t ret = nicOps->destroy(nicCtx);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("[%s] plugin destroy failed, ret[%d].", __func__, ret);
            }
        }
        delete nicOps;
        nicOps = nullptr;
    }
}

/**
 * @note 职责：NIC插件Endpoint/Channel占位子类，仅用于在EndpointMap、ChannelMap中占位，
 *         接口业务功能由NIC插件实现。
 */
class PluginEndpointHolder : public Endpoint {
public:
    explicit PluginEndpointHolder(const EndpointDesc& endpointDesc, const NicPluginEntry* pluginEntry)
        : Endpoint(endpointDesc),
          pluginEntry_(pluginEntry)
    {}
    ~PluginEndpointHolder() override { DestroyNicPluginOpsAndCtx(nicOps_, nicCtx_); }

    const NicPluginEntry* GetPluginEntry() const { return pluginEntry_; }

    HcclResult Init() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult ServerSocketListen(const uint32_t port) override
    {
        (void)port;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) override
    {
        (void)mem;
        (void)memTag;
        (void)memHandle;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult UnregisterMemory(void* memHandle) override
    {
        (void)memHandle;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen) override
    {
        (void)memHandle;
        (void)memDesc;
        (void)memDescLen;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override
    {
        (void)memDesc;
        (void)descLen;
        (void)outMem;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override
    {
        (void)memDesc;
        (void)descLen;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override
    {
        (void)memHandles;
        (void)memHandleNum;
        return HCCL_E_NOT_SUPPORT;
    }

private:
    const NicPluginEntry* pluginEntry_;
};

class PluginChannelHolder : public Channel {
public:
    explicit PluginChannelHolder(const NicPluginEntry* pluginEntry) : pluginEntry_(pluginEntry) {}
    ~PluginChannelHolder() override { DestroyNicPluginOpsAndCtx(nicOps_, nicCtx_); }

    const NicPluginEntry* GetPluginEntry() const { return pluginEntry_; }

    HcclResult Init() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override
    {
        (void)notifyNum;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override
    {
        (void)memNum;
        (void)remoteMem;
        (void)memInfos;
        return HCCL_E_NOT_SUPPORT;
    }
    ChannelStatus GetStatus() override { return ChannelStatus::FAILED; }
    HcclResult Clean() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult Resume() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override
    {
        (void)remoteNotifyIdx;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override
    {
        (void)localNotifyIdx;
        (void)timeout;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        (void)dst;
        (void)src;
        (void)len;
        (void)remoteNotifyIdx;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult Write(void* dst, const void* src, uint64_t len) override
    {
        (void)dst;
        (void)src;
        (void)len;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult Read(void* dst, const void* src, uint64_t len) override
    {
        (void)dst;
        (void)src;
        (void)len;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult ChannelFence() override { return HCCL_E_NOT_SUPPORT; }
    const HcommChannelDesc& GetChannelDesc() const override
    {
        static const HcommChannelDesc kEmptyDesc{};
        return kEmptyDesc;
    }

private:
    const NicPluginEntry* pluginEntry_;
};

} // namespace hcomm

#endif // HCOMM_NIC_PLUGIN_HOLDER_H
