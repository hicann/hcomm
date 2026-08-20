/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint.h"
#include <functional>
#include "aicpu_ts_roce_endpoint.h"
#include "cpu_roce_endpoint.h"
#include "urma_endpoint.h"
#include "ub_mem_endpoint.h"
#include "uboe_endpoint.h"
#include "ub_rtp_endpoint.h"
#include "cpu_urma_endpoint.h"
#include "aicputs_hccs_endpoint.h"
#include "hccp_nda.h"
#include "adapter_rts_common.h"
#include "rdma_handle_manager.h"
#include "proc_reged_mem_mgr_cache.h"
#include "dfx/endpoint_monitor.h"
#include "log.h"

namespace hcomm {
static bool IsSupported(const EndpointDesc& endpointDesc)
{
    bool protocolSupported = false;
    bool locTypeSupported = false;
    switch (endpointDesc.protocol) {
        case COMM_PROTOCOL_ROCE:
        case COMM_PROTOCOL_UBC_TP:
        case COMM_PROTOCOL_UB_CTP:
        case COMM_PROTOCOL_UB_MEM:
        case COMM_PROTOCOL_PCIE:
        case COMM_PROTOCOL_UBOE:
        case COMM_PROTOCOL_UB_RTP:
        case COMM_PROTOCOL_HCCS:
            protocolSupported = true;
            break;
        default:
            return false;
    }
    switch (endpointDesc.loc.locType) {
        case ENDPOINT_LOC_TYPE_DEVICE:
        case ENDPOINT_LOC_TYPE_HOST:
            locTypeSupported = true;
            break;
        default:
            return false;
    }

    return protocolSupported && locTypeSupported;
}

Endpoint::Endpoint(const EndpointDesc& endpointDesc) { endpointDesc_ = endpointDesc; }

Endpoint::~Endpoint()
{
    ReleaseEndpointMonitor(reinterpret_cast<EndpointHandle>(this));
    ReleaseCache();
    // JettyContext 由 unique_ptr 自动析构：refCount 归 0 销毁 jetty，refCount > 0 告警避免 use-after-free。
    // 控制面资源（设备上下文/注册内存）由各子类析构处理，与数据面 jetty 资源解耦。
    // SharedJettyMgr 反查记录由 HcommEndpointDestroy 在 RemoveEndpoint 前摘除（运行期单例确定存活），
    // 不在 ~Endpoint 调用，规避 g_EndpointMap 静态析构与 SharedJettyMgr 单例析构顺序不确定的风险。
}

HcclResult
Endpoint::AcquireSharedJetty(const std::function<HcclResult(SharedJettyCtx&)>& provideCtx, SharedJettyCtx& outCtx)
{
    JettyContext* ctx = GetJettyContext();
    CHK_PTR_NULL(ctx);
    return ctx->Acquire(provideCtx, outCtx);
}

HcclResult Endpoint::AcquireSharedRemoteJetty(
    const uint8_t* remoteQpKey, uint32_t keySize, bool& needImport, uint64_t& handle, void*& handlePtr, uint32_t& tpn)
{
    JettyContext* ctx = GetJettyContext();
    CHK_PTR_NULL(ctx);
    return ctx->AcquireSharedRemoteJetty(remoteQpKey, keySize, needImport, handle, handlePtr, tpn);
}

HcclResult Endpoint::PublishSharedRemoteJetty(
    const uint8_t* remoteQpKey, uint32_t keySize, uint64_t handle, void* handlePtr, uint32_t tpn)
{
    JettyContext* ctx = GetJettyContext();
    CHK_PTR_NULL(ctx);
    return ctx->PublishSharedRemoteJetty(remoteQpKey, keySize, handle, handlePtr, tpn);
}

HcclResult Endpoint::ReleaseSharedJetty()
{
    JettyContext* ctx = GetJettyContext();
    CHK_PTR_NULL(ctx);
    return ctx->Release();
}

JettyContext* Endpoint::GetJettyContext()
{
    std::call_once(jettyContextOnce_, [this] {
        jettyContext_ = std::make_unique<JettyContext>();
    });
    return jettyContext_.get();
}

HcclResult Endpoint::CreateEndpoint(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr)
{
    if (!IsSupported(endpointDesc)) {
        HCCL_ERROR(
            "[%s]endpointDesc is not supported. endpointDesc.protocol [%d] endpointDesc.loc.locType [%d].", __func__,
            endpointDesc.protocol, endpointDesc.loc.locType);
        return HCCL_E_PARA;
    }

    HCCL_INFO(
        "[%s]endpointDesc.protocol [%d] endpointDesc.loc.locType [%d].", __func__, endpointDesc.protocol,
        endpointDesc.loc.locType);

    return CreateEndpointBase(endpointDesc, endpointPtr);
}

HcclResult Endpoint::CreateEndpointBase(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr)
{
    using EndpointCreator = std::function<std::unique_ptr<Endpoint>(const EndpointDesc&)>;
    struct Entry {
        CommProtocol protocol;
        EndpointLocType locType;
        EndpointCreator creator;
    };
    static const Entry table[] = {
        {COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST,
         [](const EndpointDesc& d) {
             return std::make_unique<CpuRoceEndpoint>(d);
         }},
        {COMM_PROTOCOL_UBC_TP, ENDPOINT_LOC_TYPE_HOST,
         [](const EndpointDesc& d) {
             return std::make_unique<CpuUrmaEndpoint>(d);
         }},
        {COMM_PROTOCOL_UB_CTP, ENDPOINT_LOC_TYPE_HOST,
         [](const EndpointDesc& d) {
             return std::make_unique<CpuUrmaEndpoint>(d);
         }},
        {COMM_PROTOCOL_UBC_TP, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<UrmaEndpoint>(d);
         }},
        {COMM_PROTOCOL_UB_CTP, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<UrmaEndpoint>(d);
         }},
        {COMM_PROTOCOL_UB_MEM, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<UbMemEndpoint>(d);
         }},
        {COMM_PROTOCOL_PCIE, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<UbMemEndpoint>(d);
         }},
        {COMM_PROTOCOL_UBOE, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<UboeEndpoint>(d);
         }},
        {COMM_PROTOCOL_UB_RTP, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<UbRtpEndpoint>(d);
         }},
        {COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<AicpuTsRoceEndpoint>(d);
         }},
        {COMM_PROTOCOL_HCCS, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<AicpuTsHccsEndpoint>(d);
         }},
    };

    for (const auto& entry : table) {
        if (entry.protocol == endpointDesc.protocol && entry.locType == endpointDesc.loc.locType) {
            EXCEPTION_CATCH(endpointPtr = entry.creator(endpointDesc), return HCCL_E_PTR);
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR(
        "[%s] failed, endpointDesc.protocol [%d] and endpointDesc.loc.locType [%d] do not match.", __func__,
        endpointDesc.protocol, endpointDesc.loc.locType);
    return HCCL_E_PARA;
}

HcclResult Endpoint::CheckFeature(const EndpointDesc& endpointDesc, HcommEndpointFeatureType featureType, bool& value)
{
    if (featureType == HCOMM_ENDPOINT_FEATURE_NDA) {
        if (endpointDesc.protocol != COMM_PROTOCOL_ROCE || endpointDesc.loc.locType != ENDPOINT_LOC_TYPE_HOST) {
            HCCL_WARNING(
                "[%s] not support NDA, protocol[%d], locType[%d]", __func__, endpointDesc.protocol,
                endpointDesc.loc.locType);
            value = false;
            return HCCL_SUCCESS;
        }

        Hccl::IpAddress ipAddr{};
        CHK_RET(CommAddrToIpAddress(endpointDesc.commAddr, ipAddr));
        s32 devId = 0;
        CHK_RET(hrtGetDevice(&devId));
        u32 devPhyId = 0;
        CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));

        auto& rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
        void* rdmaHandle = static_cast<void*>(
            rdmaHandleMgr.GetByAddr(devPhyId, Hccl::LinkProtoType::RDMA, ipAddr, Hccl::PortDeploymentType::HOST_NET));
        CHK_PTR_NULL(rdmaHandle);

        s32 directFlag = 0;
        s32 ret = RaNdaGetDirectFlag(rdmaHandle, &directFlag);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] failed to get directFlag, ret[%d]", __func__, ret), HCCL_E_INTERNAL);
        value = (directFlag != DIRECT_FLAG_NOTSUPP);
        HCCL_INFO(
            "[%s] %s NDA, rdmaHandle[%p], directFlag[%d]", __func__, value ? "support" : "not support", rdmaHandle,
            directFlag);
    } else {
        HCCL_WARNING("[%s] unsupported featureType[%d]", __func__, featureType);
        value = false;
    }

    return HCCL_SUCCESS;
}

HcclResult Endpoint::AttachCache(const MemMgrCacheKey& key, std::function<std::shared_ptr<RegedMemMgr>()> creator)
{
    cacheKey_ = key;
    cacheKeepAlive_ = ProcRegedMemMgrCache::GetHolder();
    regedMemMgr_ = cacheKeepAlive_->GetOrCreate(cacheKey_, std::move(creator));
    if (regedMemMgr_ == nullptr) {
        ReleaseCache();
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

void Endpoint::ReleaseCache()
{
    if (cacheKeepAlive_ == nullptr) {
        return;
    }
    cacheKeepAlive_->Release(cacheKey_);
    cacheKeepAlive_.reset();
}

void Endpoint::AttachMonitor(s32 logicId) { monitorKeepAlive_ = EndpointMonitor::GetHolder(logicId); }

HcclResult Endpoint::RegisterToEndpointMonitor(s32 logicId, EndpointHandle handle)
{
    CHK_PRT_RET(
        monitorKeepAlive_ == nullptr, HCCL_ERROR("[Endpoint][%s] monitor not attached", __func__), HCCL_E_INTERNAL);
    return monitorKeepAlive_->RegisterToEndpointMonitor(logicId, handle);
}

void Endpoint::ReleaseEndpointMonitor(EndpointHandle handle)
{
    if (monitorKeepAlive_ == nullptr) {
        return;
    }
    monitorKeepAlive_->RemoveEpHandleFromEndpointMonitor(handle);
    monitorKeepAlive_.reset();
}
} // namespace hcomm
