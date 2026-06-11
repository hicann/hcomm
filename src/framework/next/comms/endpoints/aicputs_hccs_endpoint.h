/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPUTS_HCCS_ENDPOINT_H
#define AICPUTS_HCCS_ENDPOINT_H

#include <memory>
#include <vector>
#include <string>
#include "endpoint.h"
#include "hccl_ip_address.h"
#include "remote_ipc_rma_buffer.h"

namespace hcomm {
constexpr uint32_t AICPU_CHANNEL_DEFAULT_PORT = 16666;

/**
 * @note 职责：AICPU_TS通信引擎+HCCS协议的通信设备EndPoint，管理通信设备上下文，以及设备上的注册内存。
 */
class AicpuTsHccsEndpoint : public Endpoint {
public:
    explicit AicpuTsHccsEndpoint(const EndpointDesc &endpointDesc);

    ~AicpuTsHccsEndpoint() override;

    HcclResult Init() override;
    HcclResult ServerSocketListen(const uint32_t port) override;
    HcclResult ServerSocketStopListen(const uint32_t port) override;
    HcclResult RegisterMemory(HcommMem mem, const char *memTag, void **memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult MemoryExport(void *memHandle, void **memDesc, uint32_t *memDescLen) override;
    HcclResult MemoryImport(const void *memDesc, uint32_t descLen, HcommMem *outMem) override;
    HcclResult MemoryUnimport(const void *memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void **memHandles, uint32_t *memHandleNum) override;

    HcclResult MemoryGrant(const HcommMemGrantInfo *remoteGrantInfo) override;
    HcclResult MemoryEnableP2P(const EndpointDesc &remoteEndpointDesc);
    HcclResult MemoryDisableP2P(const EndpointDesc &remoteEndpointDesc);
    HcclResult MemoryOpenRemoteIpc();
    HcclResult MemoryCloseRemoteIpc();
    HcclResult GetRemoteIpcRmaBuffer(std::vector<HcclMem> &remoteIpcRmaBufferVec);
    HcclResult GetRemoteIpcRmaBufferEx(std::vector<HcclMemEx> &remoteIpcRmaBufferVecEx);
    HcclResult GetLocalIpcRmaBufferEx(std::vector<HcclMemEx> &localIpcRmaBufferVecEx);

private:
    u32 serverPort_{AICPU_CHANNEL_DEFAULT_PORT};
    hccl::HcclIpAddress devIpAddr_;
    bool serverListened_{false};
    HcclNetDevCtx netDevCtx_{nullptr};
};
}
#endif // AICPUTS_HCCS_ENDPOINT_H
