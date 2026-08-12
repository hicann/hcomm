/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ENDPOINT_H
#define ENDPOINT_H

#include <memory>
#include <mutex>
#include <functional>
#include <vector>
#include <string>
#include "reged_mem_mgr.h"
#include "socket/socket.h"
#include "socket_handle_manager.h"
#include "rdma_handle_manager.h"
#include "../../common/orion_adpt_utils.h"
#include "hccp_hdc_manager.h"
#include "hcomm_nic_plugin.h"

namespace hcomm {
/**
 * @note 职责：通信设备Endpoint的C++抽象接口类，管理通信设备上下文，以及设备上的注册内存。
 *       共享 Jetty 上下文也归属本类管理（"同一 EndpointHandle 共享一个 Jetty"），
 *       IS_SHARED_QUEUE=true 时，本 endpoint 下创建的 channel 复用 sharedJettyCtx_ 中的 jetty 句柄。
 */
class Endpoint {
public:
    /**
     * @brief 共享 Jetty 上下文：缓存的本地 jetty 句柄及其衍生字段，供同 endpoint 下多 channel 复用。
     */
    struct SharedJettyCtx {
        Hccl::JettyHandle handle{0};
        void* handlePtr{nullptr};
        uint32_t jettyId{0};
        uint64_t sqBuffVa{0};
        uint64_t dbAddr{0};
        uint8_t localQpKey[Hccl::HRT_UB_QP_KEY_MAX_LEN]{0};
        uint32_t keySize{0};
        uint32_t sqDepth{0};
        uint64_t tpHandle{0};
        uint32_t refCount{0};
        bool valid{false};    // 是否已填充有效 jetty
        bool creating{false}; // 是否正在首次创建中（用于并发等待）
        // 共享 SQ/CQ 的 PI/CI 索引内存（device 侧），同 endpoint 下多 channel 共用，
        // 避免各 channel 各自分配 PI/CI 指向同一 SQ 导致生产者索引无法协调前进。
        void* sqPiPtr{nullptr};
        void* sqCiPtr{nullptr};
        void* cqPiPtr{nullptr};
        void* cqCiPtr{nullptr};
        uint64_t queueIndexMemSize{0}; // 单段 PI/CI 内存字节数
        // 临时 connection 创建的 JFC 及其关联的 RDMA 句柄，由 Endpoint 在销毁共享 jetty 时统一销毁。
        // 临时 connection 走 TransferJettyOwnership 路径（releaseCb_ 为空），ReleaseResource 不销毁 JFC，
        // 需由 Endpoint 接管 ownership 防止设备资源泄漏。
        void* rdmaHandle{nullptr};
        uint64_t jfcHandle{0};
    };

    explicit Endpoint(const EndpointDesc& endpointDesc);

    virtual ~Endpoint();

    static HcclResult CreateEndpoint(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);

    virtual HcclResult Init() = 0;

    virtual HcclResult ServerSocketListen(const uint32_t port) = 0;

    virtual HcclResult ServerSocketStopListen(const uint32_t port) { return HCCL_E_NOT_SUPPORT; };
    virtual HcclResult ServerSocketGetListenPort(uint32_t* port) { return HCCL_E_NOT_SUPPORT; };

    virtual std::shared_ptr<RegedMemMgr> GetRegedMemMgr() { return regedMemMgr_; }

    void* GetRdmaHandle() { return ctxHandle_; }

    bool IsCtxHandleValid() const
    {
        if (ctxHandle_ == nullptr) {
            return false;
        }
        return Hccl::RdmaHandleManager::GetInstance().IsHandleValid(static_cast<Hccl::RdmaHandle>(ctxHandle_));
    }

    EndpointDesc GetEndpointDesc() { return endpointDesc_; }

    // 注册内存
    virtual HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) = 0;

    // 注销内存
    virtual HcclResult UnregisterMemory(void* memHandle) = 0;

    // 导出指定内存描述，用于交换
    virtual HcclResult MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen) = 0;

    // 基于内存描述，导入获得内存
    virtual HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) = 0;

    // 关闭内存
    virtual HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) = 0;

    virtual HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) = 0;

    virtual HcclResult MemoryGrant(const HcommMemGrantInfo* remoteGrantInfo) { return HCCL_SUCCESS; }

    static HcclResult CheckFeature(const EndpointDesc& endpointDesc, HcommEndpointFeatureType featureType, bool& value);

    // 获取UB异步事件
    virtual HcclResult GetAsyncEvents(uint32_t devPhyId, struct AsyncEvent events[], uint32_t& num)
    {
        (void)devPhyId;
        (void)events;
        num = 0;
        return HCCL_SUCCESS;
    }

    // ---- 共享 Jetty 管理（仅 IS_SHARED_QUEUE=true 时使用）----
    /**
     * @brief 获取或创建共享 jetty。命中复用 refCount++；未命中调 provideCtx 创建并缓存。
     * @param[in] provideCtx 创建回调（首次时调用，回调内创建 jetty 并填入 ctx）
     * @param[out] outCtx 输出的 jetty 上下文
     */
    HcclResult AcquireSharedJetty(const std::function<HcclResult(SharedJettyCtx&)>& provideCtx, SharedJettyCtx& outCtx);
    /** 释放共享 jetty 引用，refCount-- 归 0 时销毁 jetty 并清空 ctx */
    HcclResult ReleaseSharedJetty();
    // ------------------ NIC插件相关 ------------------
    void SetNicEndpointCtx(HcommNicEndpointOps* nicOps, void* nicCtx)
    {
        nicOps_ = nicOps;
        nicCtx_ = nicCtx;
    }
    HcommNicEndpointOps* GetNicOps() const { return nicOps_; }
    void* GetNicCtx() const { return nicCtx_; }

protected:
    static HcclResult CreateEndpointBase(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);
    void DestroySharedJettyRaResources(SharedJettyCtx& ctx, Hccl::RdmaHandle rdmaHandle, bool ctxValid) const;
    void FreeSharedJettyPtrs(SharedJettyCtx& ctx) const;
    void* ctxHandle_{nullptr};
    std::shared_ptr<RegedMemMgr> regedMemMgr_{};
    EndpointDesc endpointDesc_;

    // 共享 jetty 上下文及保护锁，仅 IS_SHARED_QUEUE=true 时使用
    mutable std::mutex sharedJettyMtx_;
    SharedJettyCtx sharedJettyCtx_{};
    HcommNicEndpointOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
};

} // namespace hcomm
#endif // ENDPOINT_H
