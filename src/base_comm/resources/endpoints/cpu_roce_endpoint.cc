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
#include "cpu_roce_endpoint.h"
#include "hccl/hccl_res.h"
#include "log.h"
#include "roce_mem.h"
#include "host_socket_handle_manager.h"
#include "adapter_rts_common.h"
#include "hccp_peer_manager.h"
#include "server_socket_manager.h"
#include "hccp.h"

using Hccl::HcclException;
using std::string;
using std::exception;
 
namespace hcomm {
CpuRoceEndpoint::CpuRoceEndpoint(const EndpointDesc &endpointDesc)
    : Endpoint(endpointDesc)
{
}

CpuRoceEndpoint::~CpuRoceEndpoint() noexcept
{
    std::lock_guard<std::mutex> lock(portMutex_);
    if (dynamicPort_ != HCCL_INVALID_PORT) {
        ServerSocketStopListenImpl(dynamicPort_);
    }
    dynamicPort_ = HCCL_INVALID_PORT;
}

HcclResult CpuRoceEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_HOST) {
        HCCL_INFO("[CpuRoceEndpoint][%s] CpuRoceEndpoint not support device", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));
    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    EXCEPTION_CATCH(Hccl::HccpPeerManager::GetInstance().Init(devId), return HCCL_E_INTERNAL);
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));
    auto &rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    TRY_CATCH_RETURN(ctxHandle_ = static_cast<void *>(
        rdmaHandleMgr.GetByAddr(devPhyId, Hccl::LinkProtoType::RDMA, ipAddr, Hccl::PortDeploymentType::HOST_NET)));
    CHK_PTR_NULL(ctxHandle_);
    HCCL_INFO("CpuRoceEndpoint::%s success, devPhyId[%u], ipAddr[%s], ctxHandle[%p]",
        __func__,
        devPhyId,
        ipAddr.Describe().c_str(),
        ctxHandle_);

    EXCEPTION_CATCH(regedMemMgr_ = std::make_unique<RoceRegedMemMgr>(), return HCCL_E_PARA);
    this->regedMemMgr_->rdmaHandle_ = this->ctxHandle_;
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::ServerSocketListen(const uint32_t port)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));

    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(devPhyId, type, 0, ipAddr);

    HCCL_INFO("[CpuRoceEndpoint::%s] devicePhyId[%u] ipAddress[%s]",
        __func__, devPhyId, ipAddr.Describe().c_str());

    uint32_t requestPort = port;
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, devPhyId, &requestPort));
    return HCCL_SUCCESS;
}

inline HcclResult CpuRoceEndpoint::ServerSocketStopListenImpl(const uint32_t port)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));

    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(devPhyId, type, 0, ipAddr);
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, port));

    return HCCL_SUCCESS;
}
 	 
HcclResult CpuRoceEndpoint::ServerSocketStopListen(const uint32_t port)
{
    return ServerSocketStopListenImpl(port);
}

HcclResult CpuRoceEndpoint::ServerSocketGetListenPort(uint32_t *port)
{
    std::lock_guard<std::mutex> lock(portMutex_);
    CHK_PTR_NULL(port);
    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(devPhyId, type, 0, ipAddr);

    HCCL_INFO("[CpuRoceEndpoint::%s] devicePhyId[%u] ipAddress[%s]",
        __func__, devPhyId, ipAddr.Describe().c_str());

    // 已有监听端口则直接返回
    if (dynamicPort_ != HCCL_INVALID_PORT) {
        *port = dynamicPort_;
        HCCL_INFO("[CpuRoceEndpoint::%s] already listening, return existing port[%u]", __func__, dynamicPort_);
        return HCCL_SUCCESS; 
    }
    uint32_t requestPort = 0;
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, devPhyId, &requestPort));
    if (requestPort == 0 || requestPort == HCCL_INVALID_PORT) {
        HCCL_ERROR("[CpuRoceEndpoint::%s] get listen port failed, port is invalid", __func__);
        return HCCL_E_NETWORK;
    }
    dynamicPort_ = requestPort;
    *port = dynamicPort_;
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::RegisterMemory(HcommMem mem, const char *memTag, void **memHandle)
{
    CHK_RET(this->regedMemMgr_->RegisterMemory(mem, memTag, memHandle));
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::UnregisterMemory(void* memHandle)
{
    CHK_RET(this->regedMemMgr_->UnregisterMemory(memHandle));
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::MemoryExport(void *memHandle, void **memDesc, uint32_t *memDescLen)
{
    CHK_RET(this->regedMemMgr_->MemoryExport(this->endpointDesc_, memHandle, memDesc, memDescLen));
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::MemoryImport(const void *memDesc, uint32_t descLen, HcommMem *outMem)
{
    CHK_RET(this->regedMemMgr_->MemoryImport(memDesc, descLen, outMem));
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::MemoryUnimport(const void *memDesc, uint32_t descLen)
{
    CHK_RET(this->regedMemMgr_->MemoryUnimport(memDesc, descLen));
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::GetAllMemHandles(void **memHandles, uint32_t *memHandleNum)
{
    CHK_RET(this->regedMemMgr_->GetAllMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

HcclResult CpuRoceEndpoint::GetCapabilities(Capabilities &caps)
{
    HCCL_INFO("[CpuRoceEndpoint::%s] START.", __func__);
    static constexpr uint64_t RDMA_MAX_WR_LENGTH = 1ULL * 1024 * 1024 * 1024; // 单次RDMA操作最大长度1GB
    if (!isCapabilitiesAvailable_) {
        // 待 HCCP 提供查询设备支持的最大发送消息的接口后，查询设备实际值。
        capabilities_.maxMsgSize = RDMA_MAX_WR_LENGTH;
        uint32_t ret = RaGetLbMax(this->regedMemMgr_->rdmaHandle_, &(capabilities_.lbMax));
        CHK_PRT_RET(ret != 0,
            HCCL_ERROR("[CpuRoceEndpoint::GetCapabilities][GetLbMax]errNo[0x%016llx] RaGetLbMax fail. "
            "return[%u], params: rdmaHandle[%p], lbMax[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, this->regedMemMgr_->rdmaHandle_, capabilities_.lbMax),
            HCCL_E_NETWORK);
        isCapabilitiesAvailable_ = true;
    }
    caps = capabilities_;
    HCCL_INFO("[CpuRoceEndpoint::%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}
}
