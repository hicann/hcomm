/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "endpoint_mgr.h"
#include "hccl_mem_defs.h"
#include "aicputs_hccs_endpoint.h"
#include "hccl/hccl_res.h"
#include "log.h"
#include "adapter_rts_common.h"
#include "net_dev/global_net_dev_manager.h"
#include "hccs_reged_mem_mgr.h"

using namespace hccl;

namespace hcomm {
AicpuTsHccsEndpoint::AicpuTsHccsEndpoint(const EndpointDesc &endpointDesc)
    : Endpoint(endpointDesc)
{
}

AicpuTsHccsEndpoint::~AicpuTsHccsEndpoint()
{
    try {
        (void)ServerSocketStopListen(serverPort_);
    }  catch (...) { }
    
    if (regedMemMgr_ != nullptr) {
        regedMemMgr_ = nullptr;
    }

    try {
        if (netDevCtx_ != nullptr) {
            (void)hccl::GlobalNetDevMgr::GetInstance(endpointDesc_.loc.device.devPhyId).UnRefNetDevCtx(
                NicType::VNIC_TYPE, devIpAddr_, serverPort_);
            netDevCtx_ = nullptr;
        }
    }  catch (...) { }
}

HcclResult AicpuTsHccsEndpoint::Init()
{
    HCCL_INFO("[%s]localEndpoint protocol[%d], type[%u], id[%u] locType[%u], devPhyId[%u], serverIdx[%u], "
        "superDevId[%u], superPodIdx[%u]",
        __func__, endpointDesc_.protocol, endpointDesc_.commAddr.type, endpointDesc_.commAddr.id,
        endpointDesc_.loc.locType, endpointDesc_.loc.device.devPhyId, endpointDesc_.loc.device.serverIdx,
        endpointDesc_.loc.device.superDevId, endpointDesc_.loc.device.superPodIdx);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_INFO("[AicpuTsHccsEndpoint][%s] AicpuTsHccsEndpoint not support host", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    u32 devPhyId = endpointDesc_.loc.device.devPhyId;
    uint32_t superDevId = endpointDesc_.loc.device.superDevId;
    CHK_RET(GlobalNetDevMgr::GetDeviceVnicIP(devPhyId, superDevId, devIpAddr_));
    HCCL_INFO("[AicpuTsHccsEndpoint]devPhyId[%u] superDevId[%u] devIpAddr_[%s] ",
        devPhyId, superDevId, devIpAddr_.GetReadableAddress());

    CHK_RET(hccl::GlobalNetDevMgr::GetInstance(endpointDesc_.loc.device.devPhyId).RefNetDevCtx(
        NicType::VNIC_TYPE, devIpAddr_, serverPort_, netDevCtx_));
    EXECEPTION_CATCH(regedMemMgr_ = std::make_shared<HccsRegedMemMgr>(netDevCtx_), return HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::ServerSocketListen(const uint32_t port)
{
    CHK_RET(hccl::GlobalNetDevMgr::GetInstance(endpointDesc_.loc.device.devPhyId).ServerInit(serverPort_));
    serverListened_ = true;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::ServerSocketStopListen(const uint32_t port)
{
    if (serverListened_) {
        CHK_RET(hccl::GlobalNetDevMgr::GetInstance(endpointDesc_.loc.device.devPhyId).ServerDeInit(port));
        serverListened_ = false;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::RegisterMemory(HcommMem mem, const char *memTag, void **memHandle)
{
    CHK_RET(GetRegedMemMgr()->RegisterMemory(mem, memTag, memHandle));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::UnregisterMemory(void* memHandle)
{
    CHK_RET(GetRegedMemMgr()->UnregisterMemory(memHandle));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::MemoryExport(void *memHandle, void **memDesc, uint32_t *memDescLen)
{
    CHK_RET(GetRegedMemMgr()->MemoryExport(this->endpointDesc_, memHandle, memDesc, memDescLen));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::MemoryImport(const void *memDesc, uint32_t descLen, HcommMem *outMem)
{
    CHK_RET(GetRegedMemMgr()->MemoryImport(memDesc, descLen, outMem));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::MemoryUnimport(const void *memDesc, uint32_t descLen)
{
    CHK_RET(GetRegedMemMgr()->MemoryUnimport(memDesc, descLen));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::GetAllMemHandles(void **memHandles, uint32_t *memHandleNum)
{
    CHK_RET(GetRegedMemMgr()->GetAllMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::MemoryGrant(const HcommMemGrantInfo *remoteGrantInfo)
{
    std::shared_ptr<RegedMemMgr> mgr = GetRegedMemMgr();
    CHK_PTR_NULL(mgr);
    HccsRegedMemMgr *hccsRegedMemMgr = (HccsRegedMemMgr *)mgr.get();
    CHK_RET(hccsRegedMemMgr->MemoryGrant(remoteGrantInfo));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::MemoryEnableP2P(const EndpointDesc &remoteEndpointDesc)
{
    std::shared_ptr<RegedMemMgr> mgr = GetRegedMemMgr();
    CHK_PTR_NULL(mgr);
    HccsRegedMemMgr *hccsRegedMemMgr = (HccsRegedMemMgr *)mgr.get();
    CHK_RET(hccsRegedMemMgr->MemoryEnableP2P(GetEndpointDesc(), remoteEndpointDesc));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::MemoryDisableP2P(const EndpointDesc &remoteEndpointDesc)
{
    std::shared_ptr<RegedMemMgr> mgr = GetRegedMemMgr();
    CHK_PTR_NULL(mgr);
    HccsRegedMemMgr *hccsRegedMemMgr = (HccsRegedMemMgr *)mgr.get();
    CHK_RET(hccsRegedMemMgr->MemoryDisableP2P(GetEndpointDesc(), remoteEndpointDesc));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::MemoryOpenRemoteIpc()
{
    std::shared_ptr<RegedMemMgr> mgr = GetRegedMemMgr();
    CHK_PTR_NULL(mgr);
    HccsRegedMemMgr *hccsRegedMemMgr = (HccsRegedMemMgr *)mgr.get();
    CHK_RET(hccsRegedMemMgr->MemoryOpenRemoteIpc());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::MemoryCloseRemoteIpc()
{
    std::shared_ptr<RegedMemMgr> mgr = GetRegedMemMgr();
    CHK_PTR_NULL(mgr);
    HccsRegedMemMgr *hccsRegedMemMgr = (HccsRegedMemMgr *)mgr.get();
    CHK_RET(hccsRegedMemMgr->MemoryCloseRemoteIpc());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::GetRemoteIpcRmaBuffer(std::vector<HcclMem> &remoteIpcRmaBufferVec)
{
    std::shared_ptr<RegedMemMgr> mgr = GetRegedMemMgr();
    CHK_PTR_NULL(mgr);
    HccsRegedMemMgr *hccsRegedMemMgr = (HccsRegedMemMgr *)mgr.get();
    CHK_RET(hccsRegedMemMgr->GetRemoteIpcRmaBuffer(remoteIpcRmaBufferVec));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::GetRemoteIpcRmaBufferEx(std::vector<HcclMemEx> &remoteIpcRmaBufferVecEx)
{
    std::shared_ptr<RegedMemMgr> mgr = GetRegedMemMgr();
    CHK_PTR_NULL(mgr);
    HccsRegedMemMgr *hccsRegedMemMgr = (HccsRegedMemMgr *)mgr.get();
    CHK_RET(hccsRegedMemMgr->GetRemoteIpcRmaBufferEx(remoteIpcRmaBufferVecEx));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsEndpoint::GetLocalIpcRmaBufferEx(std::vector<HcclMemEx> &localIpcRmaBufferVecEx)
{
    std::shared_ptr<RegedMemMgr> mgr = GetRegedMemMgr();
    CHK_PTR_NULL(mgr);
    HccsRegedMemMgr *hccsRegedMemMgr = (HccsRegedMemMgr *)mgr.get();
    CHK_RET(hccsRegedMemMgr->GetLocalIpcRmaBufferEx(localIpcRmaBufferVecEx));
    return HCCL_SUCCESS;
}
}