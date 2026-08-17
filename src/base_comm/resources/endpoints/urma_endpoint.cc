/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "urma_endpoint.h"
#include <algorithm>
#include "endpoint_mgr.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "urma_mem.h"
#include "proc_reged_mem_mgr_cache.h"
#include "adapter_rts_common.h"
#include "server_socket_manager.h"
#include "ra_rs_comm.h"

namespace hcomm {

UrmaEndpoint::UrmaEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UrmaEndpoint::~UrmaEndpoint() noexcept
{
    std::lock_guard<std::mutex> lock(portMutex_);
    if (dynamicPort_ != HCCL_INVALID_PORT) {
        ServerSocketStopListenImpl(dynamicPort_);
    }
    dynamicPort_ = HCCL_INVALID_PORT;
    ProcRegedMemMgrCache::GetInstance().Release(cacheKey_);
}

HcclResult UrmaEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_ERROR(
            "[UrmaEndpoint][%s] endpointDesc.loc.locType[%d] only support ENDPOINT_LOC_TYPE_DEVICE", __func__,
            endpointDesc_.loc.locType);
        return HCCL_E_PARA;
    }

    Hccl::IpAddress ipAddr{};
    HcclResult ret = CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call CommAddrToIpAddress failed");
        return ret;
    }

    u32 devPhyId;
    s32 deviceLogicId;
    ret = hrtGetDevice(&deviceLogicId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call hrtGetDevice failed, deviceLogicId[%d]", deviceLogicId);
        return ret;
    }
    Hccl::HccpHdcManager::GetInstance().Init(deviceLogicId);
    ret = hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devPhyId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call hrtGetDevicePhyIdByIndex failed, deviceLogicId[%d], devPhyId[%u]", deviceLogicId, devPhyId);
        return ret;
    }

    if (endpointDesc_.loc.device.devPhyId != devPhyId) {
        HCCL_WARNING(
            "[UrmaEndpoint][%s] endpointDesc.loc.device.devPhyId[%u] incorrect", __func__,
            endpointDesc_.loc.device.devPhyId);
        endpointDesc_.loc.device.devPhyId
            = devPhyId; // 当前endpointDesc.loc.device.devPhyId不准，暂时由查询的devPhyId赋值
    }

    auto& rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    EXCEPTION_CATCH(
        ctxHandle_ = static_cast<void*>(rdmaHandleMgr.GetByIp(endpointDesc_.loc.device.devPhyId, ipAddr)),
        return HCCL_E_PARA);
    CHK_PTR_NULL(ctxHandle_);
    HCCL_INFO(
        "%s success, devPhyId[%u], ipAddr[%s], ctxHandle[%p]", __func__, devPhyId, ipAddr.Describe().c_str(),
        ctxHandle_);

    cacheKey_ = MemMgrCacheKey{devPhyId, COMM_PROTOCOL_UB_CTP, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto& cache = ProcRegedMemMgrCache::GetInstance();
    EXCEPTION_CATCH(
        this->regedMemMgr_ = cache.GetOrCreate(
            cacheKey_,
            [this]() {
                auto m = std::make_shared<UbRegedMemMgr>();
                m->rdmaHandle_ = this->ctxHandle_;
                return m;
            }),
        return HCCL_E_INTERNAL);

    // ccu模式专用的资源分配器
    ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(deviceLogicId));
    CHK_PTR_NULL(ccuChannelCtxPool_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::ServerSocketListen(const uint32_t port)
{
    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_INFO(
            "[UrmaEndpoint][%s] endpointDesc.loc.locType[%d] skip create ServerSocket", __func__,
            endpointDesc_.loc.locType);
        return HCCL_SUCCESS;
    }

    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipaddr));

    HCCL_INFO("[UrmaEndpoint][%s] endpointDesc_.loc.device.devPhyId %u", __func__, endpointDesc_.loc.device.devPhyId);

    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort
        = Hccl::PortData(static_cast<Hccl::RankId>(endpointDesc_.loc.device.devPhyId), type, 0, ipaddr);

    HCCL_INFO(
        "[UrmaEndpoint][%s] devicePhyId[%u] localPort[%s]", __func__, endpointDesc_.loc.device.devPhyId,
        localPort.Describe().c_str());
    uint32_t requestPort = port;
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, endpointDesc_.loc.device.devPhyId, &requestPort));
    return HCCL_SUCCESS;
}

inline HcclResult UrmaEndpoint::ServerSocketStopListenImpl(const uint32_t port)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort
        = Hccl::PortData(static_cast<Hccl::RankId>(endpointDesc_.loc.device.devPhyId), type, 0, ipAddr);

    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, port));
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::ServerSocketStopListen(const uint32_t port) { return ServerSocketStopListenImpl(port); }

HcclResult UrmaEndpoint::ServerSocketGetListenPort(uint32_t* port)
{
    std::lock_guard<std::mutex> lock(portMutex_);
    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_INFO(
            "[UrmaEndpoint][%s] endpointDesc.loc.locType[%d] skip create ServerSocket", __func__,
            endpointDesc_.loc.locType);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[UrmaEndpoint][%s] endpointDesc_.loc.device.devPhyId %u", __func__, endpointDesc_.loc.device.devPhyId);
    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipaddr));

    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort
        = Hccl::PortData(static_cast<Hccl::RankId>(endpointDesc_.loc.device.devPhyId), type, 0, ipaddr);

    HCCL_INFO(
        "[UrmaEndpoint][%s] devicePhyId[%u] localPort[%s]", __func__, endpointDesc_.loc.device.devPhyId,
        localPort.Describe().c_str());

    // 已有监听端口则直接返回
    if (dynamicPort_ != HCCL_INVALID_PORT) {
        *port = dynamicPort_;
        HCCL_INFO("[UrmaEndpoint::%s] already listening, return existing port[%u]", __func__, dynamicPort_);
        return HCCL_SUCCESS;
    }
    uint32_t requestPort = 0;
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, endpointDesc_.loc.device.devPhyId, &requestPort));
    if (requestPort == 0 || requestPort == HCCL_INVALID_PORT) {
        HCCL_ERROR("[UrmaEndpoint::%s] get listen port failed, port is invalid.", __func__);
        return HCCL_E_NETWORK;
    }
    dynamicPort_ = requestPort;
    *port = dynamicPort_;
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::RegisterMemory(HcommMem mem, const char* memTag, void** memHandle)
{
    CHK_RET(this->regedMemMgr_->RegisterMemory(mem, memTag, memHandle));
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::UnregisterMemory(void* memHandle)
{
    CHK_RET(this->regedMemMgr_->UnregisterMemory(memHandle));
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    CHK_RET(this->regedMemMgr_->MemoryExport(this->endpointDesc_, memHandle, memDesc, memDescLen));
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
{
    CHK_RET(this->regedMemMgr_->MemoryImport(memDesc, descLen, outMem));
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    CHK_RET(this->regedMemMgr_->MemoryUnimport(memDesc, descLen));
    return HCCL_SUCCESS;
}

HcclResult UrmaEndpoint::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    CHK_RET(this->regedMemMgr_->GetAllMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

CcuChannelCtxPool* UrmaEndpoint::GetCcuChannelCtxPool() { return ccuChannelCtxPool_.get(); }

HcclResult UrmaEndpoint::GetAsyncEvents(uint32_t devPhyId, struct AsyncEvent events[], uint32_t& num)
{
    uint32_t interfaceVersion{0};

    int ret;
    // 对RaCtxGetAsyncEvents接口的版本检验
    ret = RaGetInterfaceVersion(devPhyId, RA_RS_CTX_GET_ASYNC_EVENTS, &interfaceVersion);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] devPhyId[%u] epHandle[%p] RaGetInterfaceVersion failed, ret[%d]", __func__, devPhyId, this, ret);
        return HCCL_E_INTERNAL;
    }

    if (interfaceVersion <= 1) {
        HCCL_ERROR(
            "[%s] devPhyId[%u] epHandle[%p] version[%u] not support", __func__, devPhyId, this, interfaceVersion);
        return HCCL_E_NOT_SUPPORT;
    }

    if (!IsCtxHandleValid()) {
        HCCL_ERROR(
            "[%s] devPhyId[%u] ctxHandle_[%p] has been invalidated, "
            "RdmaHandleManager may have DeInit this device",
            __func__, devPhyId, ctxHandle_);
        return HCCL_E_INTERNAL;
    }

    ret = RaCtxGetAsyncEvents(ctxHandle_, events, &num);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] devPhyId[%u] epHandle[%p] RaCtxGetAsyncEvents failed, ctxHandle[%p] ret[%d]", __func__, devPhyId,
            this, (void*)ctxHandle_, ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
