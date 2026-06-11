/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_p2p_channel.h"
#include "../../../endpoints/endpoint.h"
#include "orion_adpt_utils.h"
#include "hcomm_c_adpt.h"
#include "exception_handler.h"
#include "comm_mems.h"

#include "coll_alg_param.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "p2p_connection.h"

namespace hcomm {

AicpuTsP2pChannel::AicpuTsP2pChannel(EndpointHandle endpointHandle, const HcommChannelDesc &channelDesc):
    endpointHandle_(endpointHandle), channelDesc_(channelDesc) {}

AicpuTsP2pChannel::~AicpuTsP2pChannel()
{
    if (socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult AicpuTsP2pChannel::Makebufs(HcommMemHandle *memHandles, uint32_t memHandleNum,
    std::vector<std::shared_ptr<Hccl::Buffer>> &bufs) const
{
    bufs.clear();
    for (uint32_t i = 0; i < memHandleNum; ++i) {
        auto locMemInfo = reinterpret_cast<CommMemInfo *>(memHandles[i]);
        HCCL_INFO("[AicpuTsP2pChannel][%s] tag[%s]", __func__, locMemInfo->memTag);
        bufs.emplace_back(std::move(std::make_shared<Hccl::Buffer>(
            reinterpret_cast<uintptr_t>(locMemInfo->mem.addr), locMemInfo->mem.size,
            hccl::ConvertCommToHcclMemType(locMemInfo->mem.type), locMemInfo->memTag)
        ));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::ParseInputParam()
{
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();

    HCCL_INFO("[AicpuTsP2pChannel][%s] localProtocol[%d]", __func__, localEp_.protocol);

    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = reinterpret_cast<Hccl::Socket*>(channelDesc_.socket);
    notifyNum_ = channelDesc_.notifyNum;

    if (channelDesc_.exchangeAllMems) {
        HCCL_INFO("[AicpuTsP2pChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalIpcRmaBuffer> *memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(HcommMemGetAllMemHandles(
            endpointHandle_, reinterpret_cast<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AicpuTsP2pChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalIpcRmaBuffer> &localIpcRmaBuffer = memHandles[i];
            HCCL_INFO("[AicpuTsP2pChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memTag[%s].",
                __func__, i, localIpcRmaBuffer->GetBufferInfo().first,
                localIpcRmaBuffer->GetBufferInfo().second,
                localIpcRmaBuffer->GetBuf()->GetMemTag().c_str());
            bufs_.emplace_back(std::move(std::make_shared<Hccl::Buffer>(
                localIpcRmaBuffer->GetBufferInfo().first,
                localIpcRmaBuffer->GetBufferInfo().second,
                localIpcRmaBuffer->GetBuf()->GetMemTag().c_str())
            ));
        }
    } else {
        HCCL_INFO("[AicpuTsP2pChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_RET(Makebufs(channelDesc_.memHandles, channelDesc_.memHandleNum, bufs_));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildAttr()
{
    attr_.devicePhyId = localEp_.loc.device.devPhyId;
    attr_.opMode      = Hccl::OpMode::OPBASE;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildConnection()
{
    std::unique_ptr<Hccl::P2PConnection> p2pConn = nullptr;
    std::string connTag = "P2P_CHANNEL_" + std::to_string(localEp_.loc.device.devPhyId);
   
    EXECEPTION_CATCH(
        p2pConn = std::make_unique<Hccl::P2PConnection>(socket_, connTag),
        return HCCL_E_PTR
    );
    CHK_SMART_PTR_NULL(p2pConn);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(p2pConn.get());
    connections_.clear();
    connections_.push_back(std::move(p2pConn));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildNotify()
{
    localNotifies_.clear();
    commonRes_.notifyVec.clear();
    bool devUsed = true;
    for (uint32_t i = 0; i < notifyNum_; ++i) {
        std::unique_ptr<Hccl::IpcLocalNotify> notifyPtr = nullptr;
        EXECEPTION_CATCH(
            notifyPtr = std::make_unique<Hccl::IpcLocalNotify>(devUsed),
            return HCCL_E_PTR
        );
        commonRes_.notifyVec.push_back(notifyPtr.get());
        localNotifies_.push_back(std::move(notifyPtr));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildBuffer(std::vector<std::shared_ptr<Hccl::Buffer>> &bufs)
{
    bufferVecTemp_.clear();
    for (size_t i = 0; i < bufs.size(); i++) {
        std::unique_ptr<Hccl::LocalIpcRmaBuffer> bufferPtr = nullptr;
        EXECEPTION_CATCH(
            bufferPtr = std::make_unique<Hccl::LocalIpcRmaBuffer>(bufs[i]),
            return HCCL_E_PTR
        );
        bufferVecTemp_.push_back(bufferPtr.get());
        commonRes_.bufferVec.push_back(bufferPtr.get());
        localRmaBuffers_.push_back(std::move(bufferPtr));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildP2pMemTransport()
{
    const Hccl::Socket &socket = *socket_;

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));

    EXECEPTION_CATCH(
        memTransport_ = std::make_unique<Hccl::P2PTransport>(
            commonRes_, attr_, linkData, socket
        ),
        return HCCL_E_PTR
    );
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuTsP2pChannel][%s] socket ptr is NULL, rebuildSocket", __func__);

    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, ipaddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::PCIE);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(localEp_.loc.device.devPhyId), type, 0, ipaddr);
    Hccl::SocketHandle socketHandle = Hccl::SocketHandleManager::GetInstance().Create(localEp_.loc.device.devPhyId, localPort);
    EXECEPTION_CATCH(serverSocket_ = std::make_unique<Hccl::Socket>(socketHandle, ipaddr, 60001,
        ipaddr, "server", Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE), return HCCL_E_PARA);
    HCCL_INFO("[AicpuTsP2pChannel][%s] listen_socket_info[%s]", __func__, serverSocket_->Describe().c_str());
    EXECEPTION_CATCH(serverSocket_->Listen(), return HCCL_E_INTERNAL);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[AicpuTsP2pChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
    std::string socketTag = "AUTOMATIC_SOCKET_TAG";
    bool noRankId = true;
    Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, socketTag, noRankId);
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::Init()
{
    CHK_RET(ParseInputParam());
    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    CHK_RET(BuildSocket());
    CHK_RET(BuildAttr());
    CHK_RET(BuildConnection());
    CHK_RET(BuildNotify());
    localRmaBuffers_.clear();
    commonRes_.bufferVec.clear();
    CHK_RET(BuildBuffer(bufs_));
    CHK_RET(BuildP2pMemTransport());
    
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::GetNotifyNum(uint32_t *notifyNum) const
{
    *notifyNum = this->notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags)
{
    return memTransport_->GetRemoteMem(remoteMem, memNum, memTags);
}

ChannelStatus AicpuTsP2pChannel::GetStatus()
{
    ChannelStatus out = Channel::TransportStatusToChannelStatus(memTransport_->GetStatus());
    if (out == ChannelStatus::READY && socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
    }
    return out;
}

HcclResult AicpuTsP2pChannel::SetModuleDataName(Hccl::ModuleData &module, const std::string &name) const
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[SetModuleDataName] strcpy_s name %s failed", name.c_str());
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::PackOpData(std::vector<char> &data)
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetModuleDataName(dataVec[resType], "P2PTransport"));

    std::vector<char> result;
    Hccl::BinaryStream      binaryStream;
    binaryStream << memTransport_->GetUniqueIdV2();

    binaryStream.Dump(result);

    dataVec[resType].data = result;

    Hccl::AicpuResPackageHelper helper;
    data = helper.GetPackedData(dataVec);

    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::H2DResPack(std::vector<char>& buffer)
{
    CHK_RET(PackOpData(buffer));
    HCCL_INFO("[AicpuTsP2pChannel][%s] Pack Buffer data[%p], Pack Buffer size[%zu].",
        __func__, buffer.data(), buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::Clean()
{
    memTransport_.reset();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::Resume()
{
    BuildSocket();
    BuildConnection();
    BuildP2pMemTransport();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::GetUserRemoteMem(CommMem **remoteMem, char ***memTag, uint32_t *memNum)
{
    return memTransport_->GetUserRemoteMem(remoteMem, memTag, memNum);
}

HcclResult AicpuTsP2pChannel::UpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum)
{
    HCCL_WARNING("[AicpuTsP2pChannel][%s] P2PTransport does not support UpdateMemInfo.", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsP2pChannel::NotifyRecord(const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsP2pChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsP2pChannel::NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsP2pChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsP2pChannel::WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsP2pChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsP2pChannel::Write(void *dst, const void *src, uint64_t len)
{
    HCCL_INFO("[AicpuTsP2pChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsP2pChannel::Read(void *dst, const void *src, uint64_t len)
{
    HCCL_INFO("[AicpuTsP2pChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsP2pChannel::ChannelFence()
{
    HCCL_INFO("[AicpuTsP2pChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

} // namespace hcomm
