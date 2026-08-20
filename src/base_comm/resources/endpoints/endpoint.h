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
#include "proc_reged_mem_mgr_cache.h"
#include "jetty_context.h"
#include "dfx/endpoint_monitor.h"

namespace hcomm {
class EndpointMonitor;
/**
 * @note 职责：通信设备Endpoint的C++抽象接口类，管理通信设备上下文，以及设备上的注册内存。
 *       共享 Jetty 数据面资源（jetty 句柄 / QP key / PI·CI / JFC / 远端 jetty 缓存）内聚于
 *       JettyContext，Endpoint 持 unique_ptr 延迟创建，控制面职责不因共享 jetty 特性膨胀。
 *       IS_SHARED_QUEUE=true 时，本 endpoint 下创建的 channel 复用 jettyContext_ 中的 jetty 句柄。
 */
class Endpoint {
public:
    // 兼容别名：历史代码直接引用 Endpoint::SharedJettyCtx，实际为 JettyContext::Ctx 视图
    using SharedJettyCtx = JettyContext::Ctx;
    using SharedRemoteJettyCtx = JettyContext::SharedRemoteJettyCtx;

    explicit Endpoint(const EndpointDesc& endpointDesc);

    virtual ~Endpoint();

    static HcclResult CreateEndpoint(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);

    virtual HcclResult Init() = 0;

    virtual HcclResult ServerSocketListen(const uint32_t port) = 0;

    virtual HcclResult ServerSocketStopListen([[maybe_unused]] const uint32_t port) { return HCCL_E_NOT_SUPPORT; };
    virtual HcclResult ServerSocketGetListenPort([[maybe_unused]] uint32_t* port) { return HCCL_E_NOT_SUPPORT; };

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

    virtual HcclResult MemoryGrant([[maybe_unused]] const HcommMemGrantInfo* remoteGrantInfo) { return HCCL_SUCCESS; }

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

    HcclResult AcquireSharedRemoteJetty(
        const uint8_t* remoteQpKey, uint32_t keySize, bool& needImport, uint64_t& handle, void*& handlePtr,
        uint32_t& tpn);
    HcclResult PublishSharedRemoteJetty(
        const uint8_t* remoteQpKey, uint32_t keySize, uint64_t handle, void* handlePtr, uint32_t tpn);

    /** 延迟创建并获取 JettyContext（首次调用创建，后续返回已有） */
    JettyContext* GetJettyContext();

    // ------------------ NIC插件相关 ------------------
    void SetNicEndpointCtx(HcommNicEndpointOps* nicOps, void* nicCtx)
    {
        nicOps_ = nicOps;
        nicCtx_ = nicCtx;
    }
    HcommNicEndpointOps* GetNicOps() const { return nicOps_; }
    void* GetNicCtx() const { return nicCtx_; }

    // Register：AttachMonitor 拷贝 GetHolder，再用持有的指针 RegisterToEndpointMonitor。
    // Destroy / 析构走 ReleaseEndpointMonitor。未 Register 的空指针直接返回。不要再调 GetHolder()。
    void AttachMonitor(s32 logicId);
    HcclResult RegisterToEndpointMonitor(s32 logicId, EndpointHandle handle);
    void ReleaseEndpointMonitor(EndpointHandle handle);

protected:
    static HcclResult CreateEndpointBase(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);
    // Init 成功后持有 Cache GetHolder 拷贝；析构走 ReleaseCache，不要再调 GetHolder()。
    HcclResult AttachCache(const MemMgrCacheKey& key, std::function<std::shared_ptr<RegedMemMgr>()> creator);
    void ReleaseCache();
    void* ctxHandle_{nullptr};
    std::shared_ptr<RegedMemMgr> regedMemMgr_{};
    EndpointDesc endpointDesc_;
    std::once_flag jettyContextOnce_;

    // 共享 jetty 数据面资源内聚于 JettyContext，延迟创建，控制面职责不膨胀
    std::unique_ptr<JettyContext> jettyContext_{nullptr};
    HcommNicEndpointOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
    MemMgrCacheKey cacheKey_{};
    std::shared_ptr<ProcRegedMemMgrCache> cacheKeepAlive_{};
    std::shared_ptr<EndpointMonitor> monitorKeepAlive_{};
};

} // namespace hcomm
#endif // ENDPOINT_H
