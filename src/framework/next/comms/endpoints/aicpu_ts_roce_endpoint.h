/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPUTS_ROCE_ENDPOINT_H
#define AICPUTS_ROCE_ENDPOINT_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <unordered_map>
#include "hccl_mem_defs.h"
#include "endpoint.h"
#include "hccl_socket.h"

namespace hcomm {
struct AicpuTsListenSocketSlot {
    std::shared_ptr<hccl::HcclSocket> socket{};
    uint32_t refCount{0U};
};

struct SocketMapKey {
    uint32_t devicePhyId;
    uint32_t port;

    bool operator==(const SocketMapKey &other) const
    {
        return devicePhyId == other.devicePhyId && port == other.port;
    }
};

struct SocketMapKeyHash {
    size_t operator()(const SocketMapKey &k) const
    {
        return std::hash<uint32_t>()(k.devicePhyId) ^ (std::hash<uint32_t>()(k.port) << 1);
    }
};

struct AicpuTsNetDevSlot {
    HcclNetDev netDev{nullptr};
    uint32_t refCount{0U};
};

class AicpuTsRoceEndpoint : public Endpoint {
public:
    explicit AicpuTsRoceEndpoint(const EndpointDesc &endpointDesc);
    ~AicpuTsRoceEndpoint() override;

    HcclResult Init() override;

    HcclResult ServerSocketListen(const uint32_t port) override;

    HcclResult RegisterMemory(HcommMem mem, const char *memTag, void **memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult MemoryExport(void *memHandle, void **memDesc, uint32_t *memDescLen) override;
    HcclResult MemoryImport(const void *memDesc, uint32_t descLen, HcommMem *outMem) override;
    HcclResult MemoryUnimport(const void *memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void **memHandles, uint32_t *memHandleNum) override;

    HcclNetDev GetNetDev() const { return netDev_; }

    HcclResult AcceptDataSocket(uint32_t port, const std::string &tag,
        std::shared_ptr<hccl::HcclSocket> &outConnected, uint32_t acceptTimeoutMs = 0) const;

    HcclResult AddListenSocketWhiteList(uint32_t port, const std::vector<SocketWlistInfo> &wlistInfos) const;

private:
    static std::unordered_map<SocketMapKey, AicpuTsListenSocketSlot, SocketMapKeyHash> &GetServerSocketMap();
    static std::mutex &ListenSocketMapMutex();
    void ReleaseListenSocketRefs();

    static std::unordered_map<uint32_t, AicpuTsNetDevSlot> &GetNetDevMap();
    static std::mutex &NetDevMapMutex();
    HcclResult AcquireSharedNetDev(uint32_t devicePhyId, const HcclNetDevInfos &info);
    void ReleaseSharedNetDev();
    HcclResult AcquireRdmaContext(uint32_t devPhyId, const EndpointDesc &endpointDesc);

    HcclNetDev netDev_{nullptr};
    uint32_t netDevRefPhyId_{UINT32_MAX};
    std::shared_ptr<hccl::HcclSocket> serverSocket_{nullptr};
    std::vector<SocketMapKey> listenRefKeys_{};
};
}
#endif // AICPUTS_ROCE_ENDPOINT_H
