/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_mem_endpoint.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "adapter_rts_common.h"
#include "server_socket_mgr.h"
#include "ub_mem.h"
#include "proc_reged_mem_mgr_cache.h"
#include "hccl_mem_defs.h"

namespace hcomm {
UbMemEndpoint::UbMemEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UbMemEndpoint::~UbMemEndpoint() noexcept { ProcRegedMemMgrCache::GetInstance().Release(cacheKey_); }

HcclResult UbMemEndpoint::Init()
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));

    cacheKey_ = MemMgrCacheKey{devPhyId, COMM_PROTOCOL_UB_MEM, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto& cache = ProcRegedMemMgrCache::GetInstance();
    EXCEPTION_CATCH(
        regedMemMgr_ = cache.GetOrCreate(
            cacheKey_,
            []() {
                return std::make_shared<UbMemRegedMemMgr>();
            }),
        return HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::ServerSocketListen(const uint32_t port)
{
    (void)port;
    HCCL_INFO("UbMemEndpoint ServerSocketListen is not supported");
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::RegisterMemory(HcommMem mem, const char* memTag, void** memHandle)
{
    CHK_RET(this->regedMemMgr_->RegisterMemory(mem, memTag, memHandle));
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::UnregisterMemory(void* memHandle)
{
    CHK_RET(this->regedMemMgr_->UnregisterMemory(memHandle));
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::MemoryExport(
    [[maybe_unused]] void* memHandle, [[maybe_unused]] void** memDesc, [[maybe_unused]] uint32_t* memDescLen)
{
    HCCL_INFO("UbMemEndpoint MemoryExport is not supported");
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::MemoryImport(
    [[maybe_unused]] const void* memDesc, [[maybe_unused]] uint32_t descLen, [[maybe_unused]] HcommMem* outMem)
{
    HCCL_INFO("UbMemEndpoint MemoryImport is not supported");
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::MemoryUnimport([[maybe_unused]] const void* memDesc, [[maybe_unused]] uint32_t descLen)
{
    HCCL_INFO("UbMemEndpoint MemoryUnimport is not supported");
    return HCCL_SUCCESS;
}

HcclResult UbMemEndpoint::GetAllMemHandles([[maybe_unused]] void** memHandles, [[maybe_unused]] uint32_t* memHandleNum)
{
    HCCL_INFO("UbMemEndpoint GetAllMemHandles is not supported");
    return HCCL_SUCCESS;
}
} // namespace hcomm
