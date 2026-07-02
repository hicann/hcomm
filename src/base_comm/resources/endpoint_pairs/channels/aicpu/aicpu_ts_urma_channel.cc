/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_urma_channel.h"
#include "endpoint.h"
#include "orion_adpt_utils.h"
#include "hcomm_c_adpt.h"
#include "exception_handler.h"
#include "comm_mems.h"
#include "config_log.h"

// Orion
#include "adapter_rts_common.h"
#include "coll_alg_param.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "aicpu_res_package_helper.h"
#include "makebufs_helper.h"

namespace hcomm {
constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;

AicpuTsUrmaChannel::AicpuTsUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc &channelDesc):
    endpointHandle_(endpointHandle), channelDesc_(channelDesc) {}

AicpuTsUrmaChannel::~AicpuTsUrmaChannel()
{
    if (channelDesc_.socket == nullptr && socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult AicpuTsUrmaChannel::Makebufs(HcommMemHandle *memHandles, uint32_t memHandleNum,
    std::vector<std::shared_ptr<Hccl::Buffer>> &bufs)
{
    return MakebufsFromLocalRmaBuffer(memHandles, memHandleNum, bufs, "AicpuTsUrmaChannel");
}

HcclResult AicpuTsUrmaChannel::ParseInputParam() 
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    // TODO: 使用 HcommEndpointGet
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();

    HCCL_INFO("[AicpuTsUrmaChannel][%s] localProtocol[%d]", __func__, localEp_.protocol);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = reinterpret_cast<Hccl::Socket*>(channelDesc_.socket);
    notifyNum_ = channelDesc_.notifyNum;

    if (channelDesc_.exchangeAllMems) {
        // 3. Get memHandles from endpoint
        HCCL_INFO("[AicpuTsUrmaChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer> *memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(HcommMemGetAllMemHandles(
            endpointHandle_, reinterpret_cast<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer> &localUbRmaBuffer = memHandles[i];
            HCCL_INFO("[AicpuTsUrmaChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], type[%d], memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localUbRmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localUbRmaBuffer->GetSize()),
                static_cast<int>(localUbRmaBuffer->GetBuf()->GetMemType()),
                localUbRmaBuffer->GetBuf()->GetMemInfo().c_str());
            bufs_.emplace_back(std::move(std::make_shared<Hccl::Buffer>(
                localUbRmaBuffer->GetAddr(), localUbRmaBuffer->GetSize(),
                localUbRmaBuffer->GetBuf()->GetMemType(),
                localUbRmaBuffer->GetBuf()->GetMemInfo().c_str())
            ));
        }
    } else {
        // 3. 从 channelDesc 的 memHandle，获得 bufs_
        HCCL_INFO("[AicpuTsUrmaChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_RET(Makebufs(channelDesc_.memHandles, channelDesc_.memHandleNum, bufs_));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildAttr()
{
    attr_.devicePhyId = localEp_.loc.device.devPhyId;
    attr_.opMode      = Hccl::OpMode::OPBASE;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildConnection()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(localEp_, remoteEp_, channelDesc_.qos, ctx));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    std::unique_ptr<Hccl::DevUbConnection> ubConn = nullptr;
    switch (ctx.protocol) {
        case Hccl::LinkProtocol::UB_TP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbTpConnection>(rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode,
                    devUsed, Hccl::HrtUbJfcMode::STARS_POLL, Hccl::IpAddress(), Hccl::IpAddress(), ctx.qosPre),
                return HCCL_E_PTR
            );
            break;
        case Hccl::LinkProtocol::UB_CTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbCtpConnection>(rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode,
                    devUsed, Hccl::HrtUbJfcMode::STARS_POLL, Hccl::IpAddress(), Hccl::IpAddress(), ctx.qosPre),
                return HCCL_E_PTR
            );
            break;
        default:
            HCCL_ERROR("%s No LinkProtocol to match", __func__);
            break;
    }
    CHK_SMART_PTR_NULL(ubConn);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildNotify()
{
    localNotifies_.clear();
    commonRes_.notifyVec.clear();
    bool devUsed = true;
    for (uint32_t i = 0; i < notifyNum_; ++i) {
        std::unique_ptr<Hccl::UbLocalNotify> notifyPtr = nullptr;
        EXCEPTION_CATCH(
            notifyPtr = std::make_unique<Hccl::UbLocalNotify>(rdmaHandle_, devUsed),
            return HCCL_E_PTR
        );
        commonRes_.notifyVec.push_back(notifyPtr.get());
        localNotifies_.push_back(std::move(notifyPtr));
    }
    return HCCL_SUCCESS;
}

// TODO: to be deleted
HcclResult AicpuTsUrmaChannel::BuildBuffer(std::vector<std::shared_ptr<Hccl::Buffer>> &bufs)
{
    bufferVecTemp_.clear();
    for (size_t i = 0; i < bufs.size(); i++) {
        std::unique_ptr<Hccl::LocalUbRmaBuffer> bufferPtr = nullptr;
        EXCEPTION_CATCH(
            bufferPtr = std::make_unique<Hccl::LocalUbRmaBuffer>(bufs[i], rdmaHandle_),
            return HCCL_E_PTR
        );
        bufferVecTemp_.push_back(bufferPtr.get());
        commonRes_.bufferVec.push_back(bufferPtr.get());
        localRmaBuffers_.push_back(std::move(bufferPtr));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildUbMemTransport()
{
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};
    locCntNotifyRes.vec.clear();
    locCntNotifyRes.desc.clear();
    const Hccl::Socket &socket = *socket_;

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));

    bool isRecvFirst = socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    // make_unique / make_shared / release 包一层抛异常的宏
    EXCEPTION_CATCH(
        memTransport_ = std::make_unique<Hccl::UbMemTransport>(
            commonRes_, attr_, linkData, socket, rdmaHandle_, locCntNotifyRes, isRecvFirst
        ),
        return HCCL_E_PTR
    );
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuTsUrmaChannel][%s] socket ptr is NULL, rebuildSocket", __func__);
    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, ipaddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(localEp_.loc.device.devPhyId), type, 0, ipaddr);
    if (channelDesc_.role == HCOMM_SOCKET_ROLE_RESERVED) {
        Hccl::SocketHandle socketHandle
            = Hccl::SocketHandleManager::GetInstance().Create(localEp_.loc.device.devPhyId, localPort);
        EXCEPTION_CATCH(serverSocket_ = std::make_unique<Hccl::Socket>(socketHandle, ipaddr, DEFAULT_LISTENING_PORT, ipaddr, "server",
                             Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE),
            return HCCL_E_PARA);
        HCCL_INFO("[AicpuTsUrmaChannel][%s] listen_socket_info[%s]", __func__, serverSocket_->Describe().c_str());
        EXCEPTION_CATCH(serverSocket_->Listen(), return HCCL_E_INTERNAL);
        Hccl::LinkData linkData = BuildDefaultLinkData();
        CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
        std::string socketTag = (channelDesc_.channelName != nullptr)
            ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
        bool noRankId = true;
        Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, socketTag, noRankId);
        CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
    } else {
        uint16_t port = channelDesc_.port;
        if (port == 0) {
            port = DEFAULT_LISTENING_PORT;
            HCCL_INFO("[AicpuTsUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
        }
        Hccl::LinkData linkData = BuildDefaultLinkData();
        CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
        std::string socketTag = (channelDesc_.channelName != nullptr)
            ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
        bool isServer = (channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER);
        Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, port, socketTag, isServer);
        CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Init()
{
    /*
        Argue result: make_unique 配合一场捕获的宏 EXCEPTION CATCH
        Attention: const 和引用
    */
    // TODO: 处理抛异常
    s32 devLogicId;
    CHK_RET(ParseInputParam());
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    CHK_RET(StartListen());
    CHK_RET(BuildSocket());
    CHK_RET(BuildAttr());
    CHK_RET(BuildConnection());
    CHK_RET(BuildNotify());
    localRmaBuffers_.clear();
    commonRes_.bufferVec.clear();
    CHK_RET(BuildBuffer(bufs_));
    CHK_RET(BuildUbMemTransport());

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::GetNotifyNum(uint32_t *notifyNum) const
{
    *notifyNum = this->notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos)
{
    return memTransport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus AicpuTsUrmaChannel::GetStatus()
{
    ChannelStatus out = Channel::TransportStatusToChannelStatus(memTransport_->GetStatus());

    if (isFirstPrintChannelInfo_ && out == ChannelStatus::READY) {
        std::string channelInfo = "create channel info:channel handle[";
        channelInfo.append(std::to_string(reinterpret_cast<uint64_t>(this)));
        channelInfo.append("] ");
        HcclResult ret = memTransport_->Describe(channelInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsUrmaChannel][%s] Describe channel info failed, ret=%d", __func__, ret);
            out = ChannelStatus::FAILED;
        } else {
            channelInfo.append(" TA[RM]"); // 目前TA只支持RM
            HCCL_CONFIG_DEBUG(hccl::HCCL_RES, "%s", channelInfo.c_str());
        }
        isFirstPrintChannelInfo_ = false;
    }
    return out;
}

HcclResult SetModuleDataName(Hccl::ModuleData &module, const std::string &name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[SetModuleDataName] strcpy_s name %s failed", name.c_str());
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::PackOpData(std::vector<char> &data)
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetModuleDataName(dataVec[resType], "UbMemTransport"));

    std::vector<char> result;
    Hccl::BinaryStream      binaryStream;
    binaryStream << memTransport_->GetUniqueIdV2();

    binaryStream.Dump(result);

    dataVec[resType].data = result;

    Hccl::AicpuResPackageHelper helper;
    data = helper.GetPackedData(dataVec);

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::H2DResPack(std::vector<char>& buffer)
{
    CHK_RET(PackOpData(buffer));
    HCCL_INFO("[AicpuTsUrmaChannel][%s] Pack Buffer data[%p], Pack Buffer size[%zu].",
        __func__, buffer.data(), buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Clean()
{
    memTransport_.reset();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Resume()
{
    BuildSocket();
    BuildConnection();
    BuildUbMemTransport();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::UpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum)
{
    CHK_RET(Makebufs(memHandles, memHandleNum, bufsTemp));
    CHK_RET(BuildBuffer(bufsTemp));
    bufs_.insert(bufs_.end(), bufsTemp.begin(), bufsTemp.end());
    return memTransport_->UpdateMemInfo(bufferVecTemp_);
}

// 返回当前 channel 类型，供上层区分不同 channel 的能力和行为
HcommChannelKind AicpuTsUrmaChannel::GetChannelKind() const
{
    return HcommChannelKind::AICPU_TS_URMA;
}

HcclResult AicpuTsUrmaChannel::NotifyRecord(const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::Write(void *dst, const void *src, uint64_t len)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::Read(void *dst, const void *src, uint64_t len)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::ChannelFence()
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::StartListen()
{
    if (channelDesc_.role != HCOMM_SOCKET_ROLE_SERVER) {
        return HCCL_SUCCESS;
    }

    uint16_t port = channelDesc_.port;
    HCCL_INFO("[AicpuTsUrmaChannel::%s] Start. EndpointHandle[0x%llx], port[%u]", __func__, reinterpret_cast<uint64_t>(endpointHandle_), port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AicpuTsUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, port, nullptr)));
    HCCL_INFO("[AicpuTsUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

} // namespace hcomm
