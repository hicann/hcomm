/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_NIC_PLUGIN_MANAGER_H
#define HCOMM_NIC_PLUGIN_MANAGER_H

#include <stddef.h>
#include <stdint.h>

#include "hcomm_c_adpt.h"
#include "hcomm_nic_plugin.h"

namespace hcomm {

struct NicPluginEntry {
    void *soHandle;
    const HcommNicPluginInfo *info;
    HcommNicPluginCreateEndpointFunc createEndpoint;
    HcommNicPluginCreateChannelFunc createChannel;
};

struct PluginEndpointCtx {
    HcommNicEndpointOps *ops;
    void *ctx;
    const NicPluginEntry *entry;
};

struct PluginChannelCtx {
    HcommNicChannelOps *ops;
    void *ctx;
    const NicPluginEntry *entry;
};

constexpr uintptr_t HCOMM_PLUGIN_HANDLE_FLAG = (static_cast<uintptr_t>(1) << 63);

#define IS_PLUGIN_HANDLE(h) ((((uintptr_t)(h)) & ::hcomm::HCOMM_PLUGIN_HANDLE_FLAG) != 0)
#define PLUGIN_EP_CTX(h) \
    (reinterpret_cast<::hcomm::PluginEndpointCtx *>((reinterpret_cast<uintptr_t>(h)) & \
        ~::hcomm::HCOMM_PLUGIN_HANDLE_FLAG))
#define PLUGIN_CH_CTX(h) \
    (reinterpret_cast<::hcomm::PluginChannelCtx *>((static_cast<uint64_t>(h)) & \
        ~static_cast<uint64_t>(::hcomm::HCOMM_PLUGIN_HANDLE_FLAG)))
#define MAKE_PLUGIN_EP_HANDLE(p) \
    (reinterpret_cast<EndpointHandle>((reinterpret_cast<uintptr_t>(p)) | ::hcomm::HCOMM_PLUGIN_HANDLE_FLAG))
#define MAKE_PLUGIN_CH_HANDLE(p) \
    (static_cast<ChannelHandle>((reinterpret_cast<uintptr_t>(p)) | ::hcomm::HCOMM_PLUGIN_HANDLE_FLAG))

void LoadAllNicPlugins();
const NicPluginEntry *FindHostNicPlugin(CommProtocol protocol);
bool ValidatePluginInfo(const char *soPath, const HcommNicPluginInfo *info,
    HcommNicPluginCreateEndpointFunc createEndpoint, HcommNicPluginCreateChannelFunc createChannel);

HcommResult CreatePluginEndpoint(const EndpointDesc *endpoint, EndpointHandle *endpointHandle);
HcommResult DestroyPluginEndpoint(EndpointHandle endpointHandle);
HcommResult CreatePluginChannel(EndpointHandle endpointHandle, const HcommChannelDesc *channelDesc,
    ChannelHandle *channelHandle);
HcommResult DestroyPluginChannel(ChannelHandle channelHandle);

HcommResult UnsupportedPluginOp(const char *opName);
bool IsEndpointOpAvailable(const HcommNicEndpointOps *ops, size_t opOffset, size_t opSize);
bool IsChannelOpAvailable(const HcommNicChannelOps *ops, size_t opOffset, size_t opSize);

} // namespace hcomm

#endif // HCOMM_NIC_PLUGIN_MANAGER_H
