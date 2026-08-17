/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "uboe_ub_rtp_endpoint_helper.h"
#include <algorithm>
#include "log.h"

namespace hcomm {

UboeUbRtpEndpointHelper::UboeUbRtpEndpointHelper(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

HcclResult UboeUbRtpEndpointHelper::ServerSocketListen(const uint32_t port)
{
    HCCL_INFO("%s is not supported", __func__);
    return HCCL_SUCCESS;
}

HcclResult UboeUbRtpEndpointHelper::ServerSocketStopListen(const uint32_t port)
{
    HCCL_INFO("%s is not supported", __func__);
    return HCCL_SUCCESS;
}

HcclResult UboeUbRtpEndpointHelper::RegisterMemory(HcommMem mem, const char* memTag, void** memHandle)
{
    CHK_RET(regedMemMgr_->RegisterMemory(mem, memTag, memHandle));
    return HCCL_SUCCESS;
}

HcclResult UboeUbRtpEndpointHelper::UnregisterMemory(void* memHandle)
{
    CHK_RET(regedMemMgr_->UnregisterMemory(memHandle));
    return HCCL_SUCCESS;
}

HcclResult UboeUbRtpEndpointHelper::MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    CHK_RET(regedMemMgr_->MemoryExport(endpointDesc_, memHandle, memDesc, memDescLen));
    return HCCL_SUCCESS;
}

HcclResult UboeUbRtpEndpointHelper::MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
{
    CHK_RET(regedMemMgr_->MemoryImport(memDesc, descLen, outMem));
    return HCCL_SUCCESS;
}

HcclResult UboeUbRtpEndpointHelper::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    CHK_RET(regedMemMgr_->MemoryUnimport(memDesc, descLen));
    return HCCL_SUCCESS;
}

HcclResult UboeUbRtpEndpointHelper::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    CHK_RET(regedMemMgr_->GetAllMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

} // namespace hcomm
