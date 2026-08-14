/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "hcomm_result_defs.h"
#include "log.h"
#include "endpoint.h"
#include "param_check_pub.h"
#include "exception_handler.h"
#include "hcomm_res_defs.h"
#include "hcomm_res.h"
#include "hcomm_mem_alloc.h"

using namespace hcomm;

HcommResult
HcommMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem, HcommMemHandle* memHandle)
{
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    return static_cast<HcclResult>(
        endpoint->GetNicOps()->registerMemory(endpoint->GetNicCtx(), mem, memTag, reinterpret_cast<void**>(memHandle)));
}

HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
{
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    return static_cast<HcclResult>(endpoint->GetNicOps()->unregisterMemory(endpoint->GetNicCtx(), memHandle));
}

HcommResult
HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen)
{
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    return static_cast<HcclResult>(
        endpoint->GetNicOps()->memoryExport(endpoint->GetNicCtx(), memHandle, memDesc, memDescLen));
}

HcommResult HcommMemImport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen, CommMem* outMem)
{
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    return static_cast<HcclResult>(
        endpoint->GetNicOps()->memoryImport(endpoint->GetNicCtx(), memDesc, descLen, outMem));
}

HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen)
{
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    return static_cast<HcclResult>(endpoint->GetNicOps()->memoryUnimport(endpoint->GetNicCtx(), memDesc, descLen));
}

/* 暂未实现 */
HcommResult HcommMemGrant(EndpointHandle endpointHandle, const HcommMemGrantInfo* remoteGrantInfo)
{
    CHK_PTR_NULL(remoteGrantInfo);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->MemoryGrant(remoteGrantInfo));
    return HCCL_SUCCESS;
}

/* 暂未实现 */
HcommResult HcommMemRemap(const EndpointHandle endpointHandle, const CommMem* memArray, uint64_t arraySize)
{
    return HCCL_E_NOT_SUPPORT;
}

HcommResult HcommMemGetAllMemHandles(EndpointHandle endpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->GetAllMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

HcommResult HcommMemAlloc(void** ptr, size_t size) { return hcomm::MemAlloc(ptr, size); }

HcommResult HcommMemFree(void* ptr) { return hcomm::MemFree(ptr); }
