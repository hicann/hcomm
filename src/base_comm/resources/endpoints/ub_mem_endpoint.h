/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UB_MEM_ENDPOINT_H
#define UB_MEM_ENDPOINT_H

#include <memory>
#include <vector>
#include <string>
#include "endpoint.h"
#include "proc_reged_mem_mgr_cache.h"

namespace hcomm {
class UbMemEndpoint : public Endpoint {
public:
    explicit UbMemEndpoint(const EndpointDesc &endpointDesc);
    ~UbMemEndpoint() noexcept override;
    // 构造函数
    HcclResult Init() override;
    HcclResult ServerSocketListen(const uint32_t port) override;
    HcclResult RegisterMemory(HcommMem mem, const char *memTag, void **memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult MemoryExport(void *memHandle, void **memDesc, uint32_t *memDescLen) override;
    HcclResult MemoryImport(const void *memDesc, uint32_t descLen, HcommMem *outMem) override;
    HcclResult MemoryUnimport(const void *memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void **memHandles, uint32_t *memHandleNum) override;
    std::shared_ptr<RegedMemMgr> GetRegedMemMgr() override {
        return regedMemMgr_;
    }

private:
    MemMgrCacheKey cacheKey_{};
};

}

#endif // UB_MEM_ENDPOINT_H
