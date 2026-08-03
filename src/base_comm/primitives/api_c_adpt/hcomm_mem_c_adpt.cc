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
#ifdef ENABLE_EXPERIMENTAL
#include "nic_plugin_dispatcher.h"
#endif

using namespace hcomm;

HcommResult HcommMemReg(
    EndpointHandle endpointHandle, const char *memTag, const CommMem *mem, HcommMemHandle *memHandle)
{
    CHK_PTR_NULL(memHandle);
    EXCEPTION_HANDLE_BEGIN
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(memHandle);
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemReg(endpointHandle, memTag, mem, memHandle, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->RegisterMemory(*mem, memTag, reinterpret_cast<void **>(memHandle)));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
{
    CHK_PTR_NULL(memHandle);
    (void)HcommResMgrInit();
    EXCEPTION_HANDLE_BEGIN
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemUnreg(endpointHandle, memHandle, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->UnregisterMemory(memHandle));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcommResult HcommMemExport(
    EndpointHandle endpointHandle, HcommMemHandle memHandle, void **memDesc, uint32_t *memDescLen)
{
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(memDescLen);
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemExport(endpointHandle, memHandle, memDesc, memDescLen, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->MemoryExport(memHandle, memDesc, memDescLen));
    return HCCL_SUCCESS;
}

HcommResult HcommMemImport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen, CommMem *outMem)
{
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(outMem);
    CHK_PRT_RET(descLen == 0, HCCL_ERROR("[%s] descLen[0] is invalid", __func__), HCCL_E_PARA);
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemImport(endpointHandle, memDesc, descLen, outMem, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_PTR_NULL(outMem);
    CommMem importedMem{};
    CHK_RET(endpoint->MemoryImport(memDesc, descLen, &importedMem));
    *outMem = importedMem;
    return HCCL_SUCCESS;
}

HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen)
{
    CHK_PTR_NULL(memDesc);
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemUnimport(endpointHandle, memDesc, descLen, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->MemoryUnimport(memDesc, descLen));
    return HCCL_SUCCESS;
}

/* 暂未实现 */
HcommResult HcommMemGrant(EndpointHandle endpointHandle, const HcommMemGrantInfo *remoteGrantInfo)
{
    CHK_PTR_NULL(remoteGrantInfo);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->MemoryGrant(remoteGrantInfo));
    return HCCL_SUCCESS;
}

/* 暂未实现 */
HcommResult HcommMemRemap(const EndpointHandle endpointHandle, const CommMem *memArray, uint64_t arraySize)
{
    return HCCL_E_NOT_SUPPORT;
}

HcommResult HcommMemGetAllMemHandles(EndpointHandle endpointHandle, void **memHandles, uint32_t *memHandleNum)
{
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->GetAllMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

HcommResult HcommMemAlloc(void **ptr, size_t size)
{
    return hcomm::MemAlloc(ptr, size);
}

HcommResult HcommMemFree(void *ptr)
{
    return hcomm::MemFree(ptr);
}
