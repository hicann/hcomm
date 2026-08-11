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
#include <chrono>
#include <thread>
#include "aicpu_ts_roce_endpoint.h"
#include "cpu_roce_endpoint.h"
#include "urma_endpoint.h"
#include "ub_mem_endpoint.h"
#include "uboe_endpoint.h"
#include "ubg_endpoint.h"
#include "cpu_urma_endpoint.h"
#include "aicputs_hccs_endpoint.h"
#include "hccp_nda.h"
#include "adapter_rts_common.h"
#include "rdma_handle_manager.h"

namespace hcomm {
static bool IsSupported(const EndpointDesc& endpointDesc)
{
    bool protocolSupported = false;
    bool locTypeSupported = false;
    switch (endpointDesc.protocol) {
        case COMM_PROTOCOL_ROCE:
        case COMM_PROTOCOL_UBC_TP:
        case COMM_PROTOCOL_UBC_CTP:
        case COMM_PROTOCOL_UB_MEM:
        case COMM_PROTOCOL_PCIE:
        case COMM_PROTOCOL_UBOE:
        case COMM_PROTOCOL_UBG:
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

void Endpoint::DestroySharedJettyRaResources(SharedJettyCtx& ctx, Hccl::RdmaHandle rdmaHandle, bool ctxValid) const
{
    if (ctx.handle != 0) {
        if (!ctxValid) {
            HCCL_WARNING("[Endpoint][%s] skip DestroyJetty, rdmaHandle=%p invalid.", __func__, ctx.rdmaHandle);
        } else {
            Hccl::HrtRaUbDestroyJetty(ctx.handle);
            HCCL_INFO(
                "[Endpoint][%s] destroyed shared jetty, handle[%llu]", __func__,
                static_cast<unsigned long long>(ctx.handle));
        }
    }
    // 销毁临时 connection 转移过来的 JFC（共享 jetty 模式下临时 connection 不自销毁 JFC）
    if (ctx.jfcHandle != 0 && ctx.rdmaHandle != nullptr) {
        if (!ctxValid) {
            HCCL_WARNING("[Endpoint][%s] skip DestroyJfc, rdmaHandle=%p invalid.", __func__, ctx.rdmaHandle);
        } else {
            Hccl::HrtRaUbDestroyJfc(rdmaHandle, ctx.jfcHandle);
            HCCL_INFO(
                "[Endpoint][%s] destroyed shared jfc, jfcHandle[%llu]", __func__,
                static_cast<unsigned long long>(ctx.jfcHandle));
        }
    }
}

void Endpoint::FreeSharedJettyPtrs(SharedJettyCtx& ctx) const
{
    if (ctx.sqPiPtr != nullptr) {
        (void)hrtFree(ctx.sqPiPtr);
    }
    if (ctx.sqCiPtr != nullptr) {
        (void)hrtFree(ctx.sqCiPtr);
    }
    if (ctx.cqPiPtr != nullptr) {
        (void)hrtFree(ctx.cqPiPtr);
    }
    if (ctx.cqCiPtr != nullptr) {
        (void)hrtFree(ctx.cqCiPtr);
    }
}

Endpoint::~Endpoint()
{
    // 防御性清理：若仍有共享 jetty 未释放（理论上 CheckEndpointDestroy 应已拦截）。
    // refCount == 0 时可安全强制销毁；refCount > 0 表示仍有 connection 持有 jetty 句柄，
    // 强制销毁会导致 use-after-free，此时仅告警不销毁（接受泄漏以避免更严重后果）。
    if (sharedJettyCtx_.valid && sharedJettyCtx_.handle != 0) {
        if (sharedJettyCtx_.refCount == 0) {
            HCCL_WARNING(
                "[Endpoint][~Endpoint] shared jetty still valid on destroy, handle[%llu], force destroy.",
                static_cast<unsigned long long>(sharedJettyCtx_.handle));
            RdmaHandle rdmaHandle = static_cast<Hccl::RdmaHandle>(sharedJettyCtx_.rdmaHandle);
            const bool ctxValid
                = rdmaHandle != nullptr && Hccl::RdmaHandleManager::GetInstance().IsHandleValid(rdmaHandle);
            if (!ctxValid) {
                HCCL_WARNING("[Endpoint][~Endpoint] skip shared jetty/jfc destroy, rdmaHandle=%p invalid.", rdmaHandle);
            } else {
                DestroySharedJettyRaResources(sharedJettyCtx_, rdmaHandle, ctxValid);
            }
            FreeSharedJettyPtrs(sharedJettyCtx_);
        } else {
            HCCL_WARNING(
                "[Endpoint][~Endpoint] shared jetty still in use, refCount[%u], handle[%llu], skip destroy "
                "to avoid use-after-free.",
                sharedJettyCtx_.refCount, static_cast<unsigned long long>(sharedJettyCtx_.handle));
        }
        sharedJettyCtx_ = SharedJettyCtx{};
    }
}

HcclResult
Endpoint::AcquireSharedJetty(const std::function<HcclResult(SharedJettyCtx&)>& provideCtx, SharedJettyCtx& outCtx)
{
    // 第一段（持锁）：检查是否已创建或正在创建。已创建则 refCount++ 返回；未创建则标记 creating。
    while (true) {
        std::unique_lock<std::mutex> lk(sharedJettyMtx_);
        if (sharedJettyCtx_.valid) {
            sharedJettyCtx_.refCount++;
            outCtx = sharedJettyCtx_;
            HCCL_INFO(
                "[Endpoint][AcquireSharedJetty] reuse shared jetty, handle[%llu], refCount[%u]",
                static_cast<unsigned long long>(outCtx.handle), sharedJettyCtx_.refCount);
            return HCCL_SUCCESS;
        }
        if (!sharedJettyCtx_.creating) {
            // 抢占创建权
            sharedJettyCtx_.creating = true;
            break;
        }
        // 其他线程正在创建：释放锁短暂 sleep 后重新检查，避免紧密 spin 占 CPU。
        lk.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // 第二段（无锁）：执行首次创建回调（含网络建链 I/O，可能耗时数秒）。
    // 创建期间不持锁，其他线程的 Acquire 会在此循环等待，Release 不被阻塞。
    SharedJettyCtx createdCtx;
    HcclResult createRet = provideCtx(createdCtx);
    if (createRet != HCCL_SUCCESS) {
        std::lock_guard<std::mutex> lk(sharedJettyMtx_);
        sharedJettyCtx_.creating = false;
        HCCL_ERROR("[Endpoint][AcquireSharedJetty] provideCtx failed, ret[%d].", createRet);
        return createRet;
    }

    // 第三段（持锁）：写入缓存，清除 creating 标记，设置 refCount=1。
    {
        std::lock_guard<std::mutex> lk(sharedJettyMtx_);
        sharedJettyCtx_ = createdCtx;
        sharedJettyCtx_.valid = true;
        sharedJettyCtx_.creating = false;
        sharedJettyCtx_.refCount = 1;
        outCtx = sharedJettyCtx_;
    }
    HCCL_INFO(
        "[Endpoint][AcquireSharedJetty] created shared jetty, handle[%llu]",
        static_cast<unsigned long long>(outCtx.handle));
    return HCCL_SUCCESS;
}

HcclResult Endpoint::ReleaseSharedJetty()
{
    std::lock_guard<std::mutex> lk(sharedJettyMtx_);
    if (!sharedJettyCtx_.valid) {
        HCCL_WARNING("[Endpoint][ReleaseSharedJetty] shared jetty already invalid, skip release.");
        return HCCL_SUCCESS;
    }
    if (sharedJettyCtx_.refCount == 0) {
        HCCL_WARNING("[Endpoint][ReleaseSharedJetty] refCount already 0, skip release.");
        return HCCL_SUCCESS;
    }
    sharedJettyCtx_.refCount--;
    HCCL_INFO(
        "[Endpoint][ReleaseSharedJetty] release shared jetty, handle[%llu], refCount[%u]",
        static_cast<unsigned long long>(sharedJettyCtx_.handle), sharedJettyCtx_.refCount);
    if (sharedJettyCtx_.refCount == 0) {
        const auto rdmaHandle = static_cast<Hccl::RdmaHandle>(sharedJettyCtx_.rdmaHandle);
        const bool ctxValid = rdmaHandle != nullptr && Hccl::RdmaHandleManager::GetInstance().IsHandleValid(rdmaHandle);
        DestroySharedJettyRaResources(sharedJettyCtx_, rdmaHandle, ctxValid);
        FreeSharedJettyPtrs(sharedJettyCtx_);
        sharedJettyCtx_ = SharedJettyCtx{};
    }
    return HCCL_SUCCESS;
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
        {COMM_PROTOCOL_UBC_CTP, ENDPOINT_LOC_TYPE_HOST,
         [](const EndpointDesc& d) {
             return std::make_unique<CpuUrmaEndpoint>(d);
         }},
        {COMM_PROTOCOL_UBC_TP, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<UrmaEndpoint>(d);
         }},
        {COMM_PROTOCOL_UBC_CTP, ENDPOINT_LOC_TYPE_DEVICE,
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
        {COMM_PROTOCOL_UBG, ENDPOINT_LOC_TYPE_DEVICE,
         [](const EndpointDesc& d) {
             return std::make_unique<UbgEndpoint>(d);
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
} // namespace hcomm
