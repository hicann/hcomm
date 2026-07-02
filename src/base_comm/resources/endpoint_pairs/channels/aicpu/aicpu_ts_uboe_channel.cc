/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_uboe_channel.h"
#include "endpoint.h"
#include "orion_adpt_utils.h"
#include "exception_handler.h"
#include "user_remote_mem_getter.h"

// Orion
#include "adapter_rts_common.h"
#include "coll_alg_param.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "aicpu_res_package_helper.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ub_conn_dto.h"
#include "user_remote_mem_getter.h"
#include "makebufs_helper.h"

namespace hcomm {

AicpuTsUboeChannel::AicpuTsUboeChannel(EndpointHandle endpointHandle, const HcommChannelDesc &channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{
}

AicpuTsUboeChannel::~AicpuTsUboeChannel()
{
    if (channelDesc_.socket == nullptr && socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult AicpuTsUboeChannel::Makebufs(HcommMemHandle *memHandles, uint32_t memHandleNum,
    std::vector<std::shared_ptr<Hccl::Buffer>> &bufs)
{
    return MakebufsFromLocalRmaBuffer(memHandles, memHandleNum, bufs, "AicpuTsUboeChannel");
}

HcclResult AicpuTsUboeChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();

    HCCL_INFO("[AicpuTsUboeChannel][%s] localProtocol[%d]", __func__, localEp_.protocol);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = reinterpret_cast<Hccl::Socket*>(channelDesc_.socket);
    notifyNum_ = channelDesc_.notifyNum;

    if (channelDesc_.exchangeAllMems) {
        // 3. Get memHandles from endpoint
        HCCL_INFO("[AicpuTsUboeChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer> *memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(HcommMemGetAllMemHandles(
            endpointHandle_, reinterpret_cast<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AicpuTsUboeChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer> &localUbRmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localUbRmaBuffer);
            auto buf = localUbRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO("[AicpuTsUboeChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], "
                "memType[%d], memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localUbRmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localUbRmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            bufs_.emplace_back(std::move(std::make_shared<Hccl::Buffer>(
                localUbRmaBuffer->GetAddr(), localUbRmaBuffer->GetSize(),
                buf->GetMemType(), buf->GetMemInfo().c_str())
            ));
        }
    } else {
        // 3. 从 channelDesc 的 memHandle，获得 bufs_
        HCCL_INFO("[AicpuTsUboeChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_RET(Makebufs(channelDesc_.memHandles, channelDesc_.memHandleNum, bufs_));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::BuildConnection()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(localEp_, remoteEp_, channelDesc_.qos, ctx));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    HCCL_INFO("[AicpuTsUboeChannel][%s] LinkProtocol[%s], locIpv4Addr[%s], rmtIpv4Addr[%s]",
        __func__, ctx.protocol.Describe().c_str(), ctx.locAddr.Describe().c_str(), ctx.rmtAddr.Describe().c_str());
    HCCL_INFO("[AicpuTsUboeChannel][%s] locAddr_[%s], rmtAddr_[%s]",
        __func__, locAddr_.Describe().c_str(), rmtAddr_.Describe().c_str());

    std::unique_ptr<Hccl::DevUbConnection> ubConn = std::make_unique<Hccl::DevUbUboeConnection>(rdmaHandle_,
        locAddr_, rmtAddr_, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL, ctx.locAddr, ctx.rmtAddr, ctx.qosPre);
    CHK_SMART_PTR_NULL(ubConn);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));

    return HCCL_SUCCESS;
}

void AicpuTsUboeChannel::BuildConn()
{
    if (BuildConnection() != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuTsUboeChannel::%s] BuildConnection failed", __func__);
    }
}

HcclResult AicpuTsUboeChannel::BuildNotify()
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

HcclResult AicpuTsUboeChannel::BuildBuffer(std::vector<std::shared_ptr<Hccl::Buffer>> &bufs)
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

HcclResult AicpuTsUboeChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuTsUboeChannel][%s] socket ptr is NULL, rebuildSocket", __func__);

    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, ipaddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(localEp_.loc.device.devPhyId), type, 0, ipaddr);
    Hccl::SocketHandle socketHandle = Hccl::SocketHandleManager::GetInstance().Create(localEp_.loc.device.devPhyId, localPort);
    EXCEPTION_CATCH(serverSocket_ = std::make_unique<Hccl::Socket>(socketHandle, ipaddr, 60001, 
        ipaddr, "server", Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE), return HCCL_E_PARA);
    HCCL_INFO("[AicpuTsUboeChannel][%s] listen_socket_info[%s]", __func__, serverSocket_->Describe().c_str());
    EXCEPTION_CATCH(serverSocket_->Listen(), return HCCL_E_INTERNAL);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[AicpuTsUboeChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
    std::string socketTag = (channelDesc_.channelName != nullptr)
        ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
    bool noRankId = true;
    Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, socketTag, noRankId);
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
    isRecvFirst_ = socket_->GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::Init()
{
    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    CHK_RET(ParseInputParam());
    CHK_RET(BuildSocket());
    CHK_RET(BuildNotify());
    localRmaBuffers_.clear();
    commonRes_.bufferVec.clear();
    CHK_RET(BuildBuffer(bufs_));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::GetNotifyNum(uint32_t *notifyNum) const
{
    *notifyNum = this->notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<Hccl::RemoteUbRmaBuffer>> remoteMemCtx{cacheValid_, rmtBufferVec_,
        remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

bool AicpuTsUboeChannel::IsSocketReady()
{
    if (socket_ == nullptr) {
        HCCL_ERROR("[%s] socket is nullptr, please check", __func__);
        channelStatus = ChannelStatus::INVALID;
        return false;
    }

    Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
    if (socketStatus == Hccl::SocketStatus::OK) {
        channelStatus = ChannelStatus::SOCKET_OK;
        return true;
    } else if (socketStatus == Hccl::SocketStatus::TIMEOUT) {
        channelStatus = ChannelStatus::SOCKET_TIMEOUT;
        return false;
    }

    return false;
}

bool AicpuTsUboeChannel::IsResReady()
{
    for (auto &it : commonRes_.connVec) {
        if (it == nullptr) {
            Hccl::THROW<Hccl::InternalException>("[AicpuTsUboeChannel::%s] failed, connection pointer is nullptr", __func__);
        }
        Hccl::RmaConnType connType = it->GetRmaConnType();
        if (connType != Hccl::RmaConnType::UB) {
            Hccl::THROW<Hccl::InternalException>("[AicpuTsUboeChannel::%s] connection type[%s] is not ub",
                __func__, connType.Describe().c_str());
        }

        auto status = it->GetStatus();
        if (status != Hccl::RmaConnStatus::EXCHANGEABLE &&
            status != Hccl::RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("[AicpuTsUboeChannel::%s] all resources ready.", __func__);
    return true;
}

bool AicpuTsUboeChannel::IsConnsReady()
{
    for (u32 i = 0; i < connNum_; i++) {
        if (commonRes_.connVec[i]->GetStatus() != Hccl::RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("[AicpuTsUboeChannel::%s] conns are ready.", __func__);
    return true;
}

void AicpuTsUboeChannel::EidPack()
{
    Hccl::IpAddress locIpv4Addr;
    CommAddrToIpAddress(localEp_.commAddr, locIpv4Addr);
    Hccl::RdmaHandleManager::GetInstance().GetEidByIpv4Addr(locIpv4Addr, locAddr_);
    sendEidData_ = locAddr_.GetUniqueId();
    HCCL_INFO("[AicpuTsUboeChannel::%s] locIpv4Addr[%s], locAddr_[%s], sendEidData_ size[%u]",
        __func__, locIpv4Addr.Describe().c_str(), locAddr_.Describe().c_str(), sendEidData_.size());
}

void AicpuTsUboeChannel::NotifyVecPack(Hccl::BinaryStream &binaryStream)
{
    binaryStream << notifyNum_;
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeChannel::%s] pack notify size[%d]", __func__, commonRes_.notifyVec.size());
    for (auto &it : commonRes_.notifyVec) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("[AicpuTsUboeChannel::%s] pack notify pos=%u, dto %s", 
            __func__, pos, dto->Describe().c_str());
        pos++;
    }
}

void AicpuTsUboeChannel::BufferVecPack(Hccl::BinaryStream &binaryStream, std::vector<Hccl::LocalRmaBuffer *> &bufferVec)
{
    binaryStream << static_cast<u32>(bufferVec.size());
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeChannel::%s] pack buffer size[%d]", __func__, bufferVec.size());
    for (auto &it : bufferVec) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO("[AicpuTsUboeChannel::%s] pack buffer pos=%u dto %s", 
                __func__, pos, dto->Describe().c_str());
        } else { // 空的buffer，dto所有字段为0(size=0)
            Hccl::ExchangeUbBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO("[AicpuTsUboeChannel::%s] pack buffer pos=%u, dto is null %s", 
                __func__, pos, exchangeDto.Describe().c_str());
        }
        pos++;
    }
}

void AicpuTsUboeChannel::ConnVecPack(Hccl::BinaryStream &binaryStream)
{
    binaryStream << connNum_;
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeChannel::%s] pack conn size[%d]", __func__, commonRes_.connVec.size());
    for (auto &it : commonRes_.connVec) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("[AicpuTsUboeChannel::%s] pack connection pos=%u, dto %s", 
            __func__, pos, dto->Describe().c_str());
        pos++;
    }
}

void AicpuTsUboeChannel::SendDataSize()
{
    sendData_.clear();
    bufferNum_    = commonRes_.bufferVec.size(); // 需要交换的buffer数量
    connNum_      = commonRes_.connVec.size();

    HCCL_INFO("notifyNum=%u, bufferNum=%u, connNum=%u", notifyNum_, bufferNum_, connNum_);

    Hccl::BinaryStream binaryStream;
    NotifyVecPack(binaryStream);
    BufferVecPack(binaryStream, commonRes_.bufferVec);
    ConnVecPack(binaryStream);

    binaryStream.Dump(sendData_);
    u32 sendSize = sendData_.size();

    // 发送数据包尺寸
    socket_->SendAsync(reinterpret_cast<u8 *>(&sendSize), sizeof(sendSize));
    HCCL_INFO("[AicpuTsUboeChannel::%s] Send size[%u] of data success. [%zu] bytes sent.",
        __func__, sendSize, sizeof(sendSize));
}

void AicpuTsUboeChannel::RecvDataSize()
{
    // 接收数据包尺寸
    socket_->RecvAsync(reinterpret_cast<u8 *>(&recvDataSize_), sizeof(recvDataSize_));
    HCCL_INFO("[AicpuTsUboeChannel::%s] Receive Data Size", __func__);
}

void AicpuTsUboeChannel::SendExchangeData()
{
    socket_->SendAsync(reinterpret_cast<u8 *>(sendData_.data()), sendData_.size());
    HCCL_INFO("[AicpuTsUboeChannel::%s] send data, size=%llu", __func__, sendData_.size());
}

void AicpuTsUboeChannel::SendEidData()
{
    EidPack();
    socket_->SendAsync(reinterpret_cast<u8 *>(sendEidData_.data()), sendEidData_.size());
    HCCL_INFO("[AicpuTsUboeChannel::%s] send eid data, size=%llu", __func__, sendEidData_.size());
}

void AicpuTsUboeChannel::RecvExchangeData()
{
    recvData_.resize(recvDataSize_);
    socket_->RecvAsync(reinterpret_cast<u8 *>(recvData_.data()), recvData_.size());
    HCCL_INFO("[AicpuTsUboeChannel::%s] recv data", __func__);
}

void AicpuTsUboeChannel::RecvEidData()
{
    recvEidData_.resize(sendEidData_.size());
    socket_->RecvAsync(reinterpret_cast<u8 *>(recvEidData_.data()), recvEidData_.size());
    HCCL_INFO("[AicpuTsUboeChannel::%s] recv eid data, size=%llu", __func__, recvEidData_.size());
}

bool AicpuTsUboeChannel::RecvDataProcess()
{
    HCCL_INFO("RecvDataProcess: size=%llu, recvDataSize=%u", recvData_.size(), recvDataSize_);
    Hccl::BinaryStream binaryStream(recvData_);
    RmtBufferVecUnpackProc(notifyNum_, binaryStream, rmtNotifyVec_, UboeRmtBufType::NOTIFY);
    RmtBufferVecUnpackProc(bufferNum_, binaryStream, rmtBufferVec_, UboeRmtBufType::BUFFER);
    return ConnVecUnpackProc(binaryStream);
}

void AicpuTsUboeChannel::RecvEidDataProcess()
{
    RmtEidUnpackProc(rmtAddr_);
}

void AicpuTsUboeChannel::RmtEidUnpackProc(Hccl::IpAddress& rmtAddr)
{
    Hccl::IpAddress rmtEidAddr(recvEidData_);
    rmtAddr = rmtEidAddr;
    HCCL_INFO("[AicpuTsUboeChannel::%s] rmtAddr[%s]", __func__, rmtAddr.Describe().c_str());
}

void AicpuTsUboeChannel::RmtBufferVecUnpackProc(u32 locNum, Hccl::BinaryStream &binaryStream, RemoteBufferVec &bufferVec,
                                            UboeRmtBufType type)
{
    u32 rmtNum;
    binaryStream >> rmtNum;
    if (type == UboeRmtBufType::BUFFER && rmtNum > MAX_BUFFER_NUM) {
        MACRO_THROW(Hccl::InvalidParamsException,
            Hccl::StringFormat("[AicpuTsUboeChannel][RmtBufferVecUnpackProc] rmtNum[%u] exceeds limit[%u]",
            rmtNum, MAX_BUFFER_NUM));
    }

    // 允许本端和远端交换内存数量不一致
    HCCL_INFO("unpack %s, locNum=%u, rmtNum=%u", type.Describe().c_str(), locNum, rmtNum);

    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        Hccl::ExchangeUbBufferDto dto;
        dto.Deserialize(binaryStream);
        if (bufferVec.size() > pos) {
            // 对于之前已经加过的资源，无需追加
            continue;
        }
        HCCL_INFO("unpack %s pos=%u, dto %s", type.Describe().c_str(), pos, dto.Describe().c_str());
        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            bufferVec.push_back(nullptr);
        } else { // size非0，则构造一个remote buffer
            bufferVec.push_back(std::make_unique<Hccl::RemoteUbRmaBuffer>(rdmaHandle_, dto));
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, bufferVec.back()->Describe().c_str());
        }
    }
}

bool AicpuTsUboeChannel::ConnVecUnpackProc(Hccl::BinaryStream &binaryStream)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn connNum=%u, rmtConnNum=%u", connNum_, rmtConnNum);
    if (connNum_ != rmtConnNum) {
        MACRO_THROW(Hccl::InvalidParamsException,
                    Hccl::StringFormat("connNum=%u is not equal to rmtConnNum=%u", connNum_, rmtConnNum));
    }

    bool result = false; // 不需要发送 finish
    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        Hccl::ExchangeUbConnDto rmtDto;
        rmtDto.Deserialize(binaryStream);
        HCCL_INFO("unpack connection pos=%u dto %s", pos, rmtDto.Describe().c_str());
        if (commonRes_.connVec[i]->GetStatus() != Hccl::RmaConnStatus::READY) {
            HCCL_INFO("parse and import pos=%u, rmt dto to connection[%s]", pos,
                       commonRes_.connVec[i]->Describe().c_str());
            commonRes_.connVec[i]->ParseRmtExchangeDto(rmtDto);
            commonRes_.connVec[i]->ImportRmtDto();
            result = true; // connection 建链，需要发送finish
        }
    }
    return result;
}

void AicpuTsUboeChannel::SendFinish()
{
    HCCL_INFO("start send Finish Msg [%s]", FINISH_MSG);
    sendFinishMsg_ = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    socket_->SendAsync(reinterpret_cast<u8 *>(sendFinishMsg_.data()), FINISH_MSG_SIZE);
    HCCL_INFO("end send Finish Msg [%s]", FINISH_MSG);
}

void AicpuTsUboeChannel::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    HCCL_INFO("start recv Finish Msg [%s]", FINISH_MSG);
    socket_->RecvAsync(reinterpret_cast<u8 *>(recvFinishMsg_.data()), FINISH_MSG_SIZE);
    HCCL_INFO("end recv Finish Msg [%s]", FINISH_MSG);
}

void AicpuTsUboeChannel::HandleProcessData()
{
    if (RecvDataProcess()) {
        uboeStatus = UboeStatus::SEND_FIN;
    } else {
        channelStatus = ChannelStatus::READY;
        uboeStatus = UboeStatus::READY;
    }
}

void AicpuTsUboeChannel::ProcessUboeState()
{
    auto SetState = [this](UboeStatus next, ChannelStatus ch) { this->uboeStatus = next; this->channelStatus = ch; };

    switch (uboeStatus) {
        case UboeStatus::INIT:
            SetState(UboeStatus::SEND_EID, ChannelStatus::SOCKET_OK);
            break;
        case UboeStatus::SEND_EID:
            SendEidData(); SetState(UboeStatus::RECV_EID, channelStatus);
            break;
        case UboeStatus::RECV_EID:
            RecvEidData(); SetState(UboeStatus::PROCESS_EID_DATA, channelStatus);
            break;
        case UboeStatus::PROCESS_EID_DATA:
            RecvEidDataProcess(); SetState(UboeStatus::BUILD_CONN, channelStatus);
            break;
        case UboeStatus::BUILD_CONN:
            BuildConn(); SetState(UboeStatus::SEND_SIZE, channelStatus);
            break;
        case UboeStatus::SEND_SIZE:
            if (IsResReady()) { SendDataSize(); SetState(UboeStatus::RECV_SIZE, channelStatus); }
            break;
        case UboeStatus::RECV_SIZE:
            RecvDataSize(); SetState(isRecvFirst_ ? UboeStatus::RECV_DATA : UboeStatus::SEND_DATA, channelStatus);
            break;
        case UboeStatus::SEND_DATA:
            SendExchangeData(); SetState(isRecvFirst_ ? UboeStatus::PROCESS_DATA : UboeStatus::RECV_DATA, channelStatus);
            break;
        case UboeStatus::RECV_DATA:
            RecvExchangeData(); SetState(isRecvFirst_ ? UboeStatus::SEND_DATA : UboeStatus::PROCESS_DATA, channelStatus);
            break;
        case UboeStatus::PROCESS_DATA:
            HandleProcessData();
            break;
        case UboeStatus::SEND_FIN:
            if (IsConnsReady()) { SendFinish(); SetState(UboeStatus::RECV_FIN, channelStatus); }
            break;
        case UboeStatus::RECV_FIN:
            RecvFinish(); SetState(UboeStatus::SET_READY, channelStatus);
            break;
        case UboeStatus::SET_READY:
            channelStatus = ChannelStatus::READY; SetState(UboeStatus::READY, ChannelStatus::READY);
            break;
        default:
            break;
    }
}

ChannelStatus AicpuTsUboeChannel::GetStatus()
{
    if (channelStatus == ChannelStatus::READY) {
        return channelStatus;
    }
    if (channelStatus == ChannelStatus::INIT) uboeStatus = UboeStatus::INIT;

    if (!IsSocketReady()) return channelStatus;

    ProcessUboeState();

    return channelStatus;
}

HcclResult SetUboeModuleDataName(Hccl::ModuleData &module, const std::string &name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[SetModuleDataName] strcpy_s name %s failed", name.c_str());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

std::vector<char> AicpuTsUboeChannel::GetNotifyUniqueIds()
{
    HCCL_INFO("start packing all notify uniqueIds");
    std::vector<char> result(0);
    for (auto &it : commonRes_.notifyVec) {
        HCCL_INFO("AicpuTsUboeChannel Notify %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeChannel::GetSingleRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId) const
{
    Hccl::BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << tokenId;
    binaryStream << tokenValue;
    binaryStream << notifyId;
    HCCL_INFO("AicpuTsUboeChannel RmtBuffer[addr=0x%llx, size=0x%llx, notifyId=%u]", addr, size, notifyId);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> AicpuTsUboeChannel::GetRmtBufferUniqueIds(RemoteBufferVec &bufferVec, UboeRmtBufType type) const
{
    HCCL_INFO("start packing all remote buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto &it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue(),
                it->GetNotifyId());
            HCCL_INFO("AicpuTsUboeChannel::GetRmtBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
            HCCL_INFO("AicpuTsUboeChannel::GetRmtBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeChannel::GetLocBufferUniqueIds(LocalBufferVec &bufferVec, UboeRmtBufType type) const
{
    HCCL_INFO("start packing all local buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto &it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue(),
                UINT32_MAX);
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeChannel::GetConnUniqueIds()
{
    HCCL_INFO("start packing all conn uniqueIds");
    std::vector<char> result(0);
    for (auto &it : commonRes_.connVec) {
        HCCL_INFO("[AicpuTsUboeChannel::%s] conn[%s]", __func__, it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeChannel::GetUniqueIdV2()
{
    if (channelStatus != ChannelStatus::READY) {
        MACRO_THROW(Hccl::InternalException, Hccl::StringFormat("channel status[%d] is not ready[%d], please check.",
            channelStatus, ChannelStatus::READY));
    }
    u32 type = static_cast<u32>(Hccl::TransportType::UB);
    Hccl::BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum_;
    binaryStream << bufferNum_;
    binaryStream << static_cast<u32>(rmtBufferVec_.size());
    binaryStream << connNum_;
 
    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;
 
    auto rmtNotifyUniqueIds = GetRmtBufferUniqueIds(rmtNotifyVec_, UboeRmtBufType::NOTIFY);
    binaryStream << rmtNotifyUniqueIds;
 
    for (auto &it : commonRes_.bufferVec) {
        locBufferVec_.emplace_back(reinterpret_cast<Hccl::LocalUbRmaBuffer *>(it));
    }
 
    auto locBufferUniqueIds = GetLocBufferUniqueIds(locBufferVec_, UboeRmtBufType::BUFFER);
    binaryStream << locBufferUniqueIds;
 
    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds(rmtBufferVec_, UboeRmtBufType::BUFFER);
    binaryStream << rmtBufferUniqueIds;
 
    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;
 
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

HcclResult AicpuTsUboeChannel::H2DResPack(std::vector<char>& buffer)
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetUboeModuleDataName(dataVec[resType], "AicpuTsUboeChannel"));

    std::vector<char> result;
    Hccl::BinaryStream      binaryStream;
    binaryStream << GetUniqueIdV2();

    binaryStream.Dump(result);

    dataVec[resType].data = result;

    Hccl::AicpuResPackageHelper helper;
    buffer = helper.GetPackedData(dataVec);
    HCCL_INFO("[AicpuTsUboeChannel][%s] Pack Buffer data[%p], Pack Buffer size[%zu].",
        __func__, buffer.data(), buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::Clean()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::Resume()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::NotifyRecord(const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUboeChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeChannel::NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsUboeChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeChannel::WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUboeChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeChannel::Write(void *dst, const void *src, uint64_t len)
{
    HCCL_INFO("[AicpuTsUboeChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeChannel::Read(void *dst, const void *src, uint64_t len)
{
    HCCL_INFO("[AicpuTsUboeChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeChannel::ChannelFence()
{
    HCCL_INFO("[AicpuTsUboeChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

} // namespace hcomm
