/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILTIN_ENDPOINT_OPS_H
#define BUILTIN_ENDPOINT_OPS_H

#include "hcomm_nic_plugin.h"
#include "endpoint.h"
#include "exception_handler.h"
#include "hcomm_c_adpt_common.h"

// CreateBuiltinEndpoint 已调用 endpointPtr->Init()，此处 no-op。
inline int32_t BuiltinEndpointInit(void* ctx)
{
    (void)ctx;
    return HCCL_SUCCESS;
}

// endpoint 生命周期由 g_EndpointMap 的 unique_ptr 管理，此处 no-op。
inline int32_t BuiltinEndpointDestroy(void* ctx)
{
    (void)ctx;
    return HCCL_SUCCESS;
}

inline int32_t BuiltinRegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle)
{
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(handle);
    EXCEPTION_HANDLE_BEGIN(void) HcommResMgrInit();
    EndpointHandle epHandle = reinterpret_cast<EndpointHandle>(ctx);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, epHandle);
    auto endpoint = GetEndpointMap().GetEndpoint(epHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, epHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->RegisterMemory(*mem, tag, handle));

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

inline int32_t BuiltinUnregisterMemory(void* ctx, void* handle)
{
    CHK_PTR_NULL(handle);
    (void)HcommResMgrInit();
    EXCEPTION_HANDLE_BEGIN
    EndpointHandle epHandle = reinterpret_cast<EndpointHandle>(ctx);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, epHandle);
    auto endpoint = GetEndpointMap().GetEndpoint(epHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, epHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->UnregisterMemory(handle));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

inline int32_t BuiltinMemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen)
{
    CHK_PTR_NULL(handle);
    CHK_PTR_NULL(desc);
    CHK_PTR_NULL(descLen);
    (void)HcommResMgrInit();
    EndpointHandle epHandle = reinterpret_cast<EndpointHandle>(ctx);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, epHandle);
    auto endpoint = GetEndpointMap().GetEndpoint(epHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, epHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->MemoryExport(handle, desc, descLen));
    return HCCL_SUCCESS;
}

inline int32_t BuiltinMemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem)
{
    CHK_PTR_NULL(desc);
    CHK_PTR_NULL(outMem);
    CHK_PRT_RET(descLen == 0, HCCL_ERROR("[%s] descLen[0] is invalid", __func__), HCCL_E_PARA);
    (void)HcommResMgrInit();
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(ctx);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, handle);
    auto endpoint = GetEndpointMap().GetEndpoint(handle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, handle),
        HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CommMem importedMem{};
    CHK_RET(endpoint->MemoryImport(desc, descLen, &importedMem));
    *outMem = importedMem;
    return HCCL_SUCCESS;
}

inline int32_t BuiltinMemoryUnimport(void* ctx, const void* desc, uint32_t descLen)
{
    CHK_PTR_NULL(desc);
    (void)HcommResMgrInit();
    EndpointHandle epHandle = reinterpret_cast<EndpointHandle>(ctx);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, epHandle);
    auto endpoint = GetEndpointMap().GetEndpoint(epHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, epHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->MemoryUnimport(desc, descLen));
    return HCCL_SUCCESS;
}

inline int32_t BuiltinGetListenPort(void* ctx, uint32_t* port)
{
    CHK_PTR_NULL(port);
    (void)HcommResMgrInit();
    EndpointHandle epHandle = reinterpret_cast<EndpointHandle>(ctx);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, epHandle);
    auto endpoint = GetEndpointMap().GetEndpoint(epHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, epHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    return endpoint->ServerSocketGetListenPort(port);
}

inline HcommNicEndpointOps g_BuiltinEndpointOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    BuiltinEndpointInit,     // init
    BuiltinEndpointDestroy,  // destroy
    BuiltinRegisterMemory,   // registerMemory
    BuiltinUnregisterMemory, // unregisterMemory
    BuiltinMemoryExport,     // memoryExport
    BuiltinMemoryImport,     // memoryImport
    BuiltinMemoryUnimport,   // memoryUnimport
    BuiltinGetListenPort,    // getListenPort
};

#endif // BUILTIN_ENDPOINT_OPS_H
