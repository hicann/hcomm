/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_uboe_ubg_channel_helper.h"
#include "endpoint.h"
#include "orion_adpt_utils.h"

// Orion
#include "virtual_topo.h"
#include "dev_capability.h"
#include "dev_buffer.h"
#include "aicpu_res_package_helper.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ub_conn_dto.h"
#include "user_remote_mem_getter.h"
#include "makebufs_helper.h"

namespace hcomm {

constexpr u32 SERVER_LISTEN_PORT = 60001;

AicpuTsUboeUbgChannelHelper::AicpuTsUboeUbgChannelHelper(
    EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

AicpuTsUboeUbgChannelHelper::~AicpuTsUboeUbgChannelHelper()
{
    if (channelDesc_.socket == nullptr && socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult AicpuTsUboeUbgChannelHelper::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();

    HCCL_INFO("[AicpuTsUboeUbgChannelHelper][%s] localProtocol[%d]", __func__, localEp_.protocol);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = reinterpret_cast<Hccl::Socket*>(channelDesc_.socket);
    notifyNum_ = channelDesc_.notifyNum;
    commonRes_.bufferVec.clear();

    if (channelDesc_.exchangeAllMems) {
        // 3. Get memHandles from endpoint
        HCCL_INFO("[AicpuTsUboeUbgChannelHelper][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, reinterpret_cast<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AicpuTsUboeUbgChannelHelper][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer>& localUbRmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localUbRmaBuffer);
            Hccl::Buffer* buf = localUbRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[AicpuTsUboeUbgChannelHelper][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], "
                "memType[%d], memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localUbRmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localUbRmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            commonRes_.bufferVec.push_back(localUbRmaBuffer.get());
        }
    } else {
        // 3. 从 channelDesc 的 memHandle 填充 commonRes_.bufferVec
        HCCL_INFO(
            "[AicpuTsUboeUbgChannelHelper][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_RET(MakeRmaBufferVecFromMemHandles(
            channelDesc_.memHandles, channelDesc_.memHandleNum, commonRes_.bufferVec, "AicpuTsUboeUbgChannelHelper"));
    }

    return HCCL_SUCCESS;
}

void AicpuTsUboeUbgChannelHelper::BuildConn()
{
    if (BuildConnection() != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuTsUboeUbgChannelHelper::%s] BuildConnection failed", __func__);
    }
}

HcclResult AicpuTsUboeUbgChannelHelper::BuildNotify()
{
    localNotifies_.clear();
    commonRes_.notifyVec.clear();
    bool devUsed = true;
    for (uint32_t i = 0; i < notifyNum_; ++i) {
        std::unique_ptr<Hccl::UbLocalNotify> notifyPtr = nullptr;
        EXCEPTION_CATCH(notifyPtr = std::make_unique<Hccl::UbLocalNotify>(rdmaHandle_, devUsed), return HCCL_E_PTR);
        commonRes_.notifyVec.push_back(notifyPtr.get());
        localNotifies_.push_back(std::move(notifyPtr));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbgChannelHelper::BuildDrainResource()
{
    // 申请创建channel drain阻塞等待的相关资源
    // 申请notify作为read的落点
    bool devUsed = true;
    EXCEPTION_CATCH(drainNotify_ = std::make_unique<Hccl::UbLocalNotify>(rdmaHandle_, devUsed), return HCCL_E_PTR);
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper][%s] drainNotify created: %s", __func__, drainNotify_->Describe().c_str());

    // 常量1内存供远端读取
    u32 notifySize = Hccl::DevCapability::GetInstance().GetNotifySize();

    std::shared_ptr<Hccl::DevBuffer> constMem;
    EXCEPTION_CATCH(constMem = std::make_shared<Hccl::DevBuffer>(notifySize), return HCCL_E_PTR);

    Hccl::HrtMemcpy(
        reinterpret_cast<void*>(constMem->GetAddr()), constMem->GetSize(), &NORMAL_NOTIFY_VAL,
        sizeof(NORMAL_NOTIFY_VAL), Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);

    EXCEPTION_CATCH(drainBuffer_ = std::make_unique<Hccl::LocalUbRmaBuffer>(constMem, rdmaHandle_), return HCCL_E_PTR);
    HCCL_INFO(
        "[AicpuTsUboeUbgChannelHelper][%s] drain buffer created: addr[0x%llx], size[%zu]", __func__,
        static_cast<unsigned long long>(drainBuffer_->GetAddr()), drainBuffer_->GetSize());

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbgChannelHelper::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper][%s] socket ptr is NULL, rebuildSocket", __func__);

    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, ipaddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(localEp_.loc.device.devPhyId), type, 0, ipaddr);
    Hccl::SocketHandle socketHandle
        = Hccl::SocketHandleManager::GetInstance().Create(localEp_.loc.device.devPhyId, localPort);
    EXCEPTION_CATCH(
        serverSocket_ = std::make_unique<Hccl::Socket>(
            socketHandle, ipaddr, SERVER_LISTEN_PORT, ipaddr, "server", Hccl::SocketRole::SERVER,
            Hccl::NicType::DEVICE_NIC_TYPE),
        return HCCL_E_PARA);
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper][%s] listen_socket_info[%s]", __func__, serverSocket_->Describe().c_str());
    EXCEPTION_CATCH(serverSocket_->Listen(), return HCCL_E_INTERNAL);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper][%s] built linkData: %s", __func__, linkData.Describe().c_str());
    std::string socketTag
        = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
    bool noRankId = true;
    Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, socketTag, noRankId);
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
    isRecvFirst_ = socket_->GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbgChannelHelper::GetNotifyNum(uint32_t* notifyNum) const
{
    *notifyNum = this->notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbgChannelHelper::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<Hccl::RemoteUbRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtBufferVec_, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

bool AicpuTsUboeUbgChannelHelper::IsSocketReady()
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

bool AicpuTsUboeUbgChannelHelper::IsResReady()
{
    for (auto& it : commonRes_.connVec) {
        if (it == nullptr) {
            Hccl::THROW<Hccl::InternalException>(
                "[AicpuTsUboeUbgChannelHelper::%s] failed, connection pointer is nullptr", __func__);
        }
        Hccl::RmaConnType connType = it->GetRmaConnType();
        if (connType != Hccl::RmaConnType::UB) {
            Hccl::THROW<Hccl::InternalException>(
                "[AicpuTsUboeUbgChannelHelper::%s] connection type[%s] is not ub", __func__,
                connType.Describe().c_str());
        }

        auto status = it->GetStatus();
        if (status != Hccl::RmaConnStatus::EXCHANGEABLE && status != Hccl::RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] all resources ready.", __func__);
    return true;
}

bool AicpuTsUboeUbgChannelHelper::IsConnsReady()
{
    for (u32 i = 0; i < connNum_; i++) {
        if (commonRes_.connVec[i]->GetStatus() != Hccl::RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] conns are ready.", __func__);
    return true;
}

void AicpuTsUboeUbgChannelHelper::NotifyVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << notifyNum_;
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] pack notify size[%d]", __func__, commonRes_.notifyVec.size());
    for (auto& it : commonRes_.notifyVec) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO(
            "[AicpuTsUboeUbgChannelHelper::%s] pack notify pos=%u, dto %s", __func__, pos, dto->Describe().c_str());
        pos++;
    }
}

void AicpuTsUboeUbgChannelHelper::BufferVecPack(
    Hccl::BinaryStream& binaryStream, std::vector<Hccl::LocalRmaBuffer*>& bufferVec)
{
    binaryStream << static_cast<u32>(bufferVec.size());
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] pack buffer size[%d]", __func__, bufferVec.size());
    for (auto& it : bufferVec) {
        binaryStream << pos;
        if (it != nullptr) {
            std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO(
                "[AicpuTsUboeUbgChannelHelper::%s] pack buffer pos=%u dto %s", __func__, pos, dto->Describe().c_str());
        } else {
            Hccl::ExchangeUbBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO(
                "[AicpuTsUboeUbgChannelHelper::%s] pack buffer pos=%u, dto is null %s", __func__, pos,
                exchangeDto.Describe().c_str());
        }
        pos++;
    }
}

void AicpuTsUboeUbgChannelHelper::DrainBufferPack(Hccl::BinaryStream& binaryStream)
{
    // 只需交换常量buffer信息供对端读
    HCCL_INFO("start pack drain buffer");
    if (drainBuffer_ != nullptr) { // 非空的buffer，从buffer中获取 dto
        std::unique_ptr<Hccl::Serializable> dto = drainBuffer_->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] pack drain buffer dto %s", __func__, dto->Describe().c_str());
    } else { // 空的buffer，dto所有字段为0(size=0)
        Hccl::ExchangeUbBufferDto exchangeDto;
        exchangeDto.Serialize(binaryStream);
        HCCL_INFO(
            "[AicpuTsUboeUbgChannelHelper::%s] pack drain buffer dto is null %s", __func__,
            exchangeDto.Describe().c_str());
    }
}

void AicpuTsUboeUbgChannelHelper::ConnVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << connNum_;
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] pack conn size[%d]", __func__, commonRes_.connVec.size());
    for (auto& it : commonRes_.connVec) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO(
            "[AicpuTsUboeUbgChannelHelper::%s] pack connection pos=%u, dto %s", __func__, pos, dto->Describe().c_str());
        pos++;
    }
}

void AicpuTsUboeUbgChannelHelper::SendDataSize()
{
    sendData_.clear();
    bufferNum_ = commonRes_.bufferVec.size();
    connNum_ = commonRes_.connVec.size();

    HCCL_INFO("notifyNum=%u, bufferNum=%u, connNum=%u", notifyNum_, bufferNum_, connNum_);

    Hccl::BinaryStream binaryStream;
    NotifyVecPack(binaryStream);
    BufferVecPack(binaryStream, commonRes_.bufferVec);
    DrainBufferPack(binaryStream);
    ConnVecPack(binaryStream);

    binaryStream.Dump(sendData_);
    u32 sendSize = sendData_.size();

    // 发送数据包尺寸
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    HCCL_INFO(
        "[AicpuTsUboeUbgChannelHelper::%s] Send size[%u] of data success. [%zu] bytes sent.", __func__, sendSize,
        sizeof(sendSize));
}

void AicpuTsUboeUbgChannelHelper::RecvDataSize()
{
    // 接收数据包尺寸
    socket_->RecvAsync(reinterpret_cast<u8*>(&recvDataSize_), sizeof(recvDataSize_));
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] Receive Data Size", __func__);
}

void AicpuTsUboeUbgChannelHelper::SendExchangeData()
{
    socket_->SendAsync(sendData_.data(), sendData_.size());
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] send data, size=%llu", __func__, sendData_.size());
}

void AicpuTsUboeUbgChannelHelper::RecvExchangeData()
{
    recvData_.resize(recvDataSize_);
    socket_->RecvAsync(reinterpret_cast<u8*>(recvData_.data()), recvData_.size());
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] recv data", __func__);
}

bool AicpuTsUboeUbgChannelHelper::RecvDataProcess()
{
    HCCL_INFO("RecvDataProcess: size=%llu, recvDataSize=%u", recvData_.size(), recvDataSize_);
    Hccl::BinaryStream binaryStream(recvData_);
    RmtBufferVecUnpackProc(notifyNum_, binaryStream, rmtNotifyVec_, UboeRmtBufType::NOTIFY);
    RmtBufferVecUnpackProc(bufferNum_, binaryStream, rmtBufferVec_, UboeRmtBufType::BUFFER);
    RmtDrainBufferUnpackProc(binaryStream);
    return ConnVecUnpackProc(binaryStream);
}

void AicpuTsUboeUbgChannelHelper::RmtBufferVecUnpackProc(
    u32 locNum, Hccl::BinaryStream& binaryStream, RemoteBufferVec& bufferVec, UboeRmtBufType type)
{
    u32 rmtNum;
    binaryStream >> rmtNum;
    if (type == UboeRmtBufType::BUFFER && rmtNum > MAX_BUFFER_NUM) {
        MACRO_THROW(
            Hccl::InvalidParamsException,
            Hccl::StringFormat(
                "[AicpuTsUboeUbgChannelHelper][RmtBufferVecUnpackProc] rmtNum[%u] exceeds limit[%u]", rmtNum,
                MAX_BUFFER_NUM));
    }

    HCCL_INFO("unpack %s, locNum=%u, rmtNum=%u", type.Describe().c_str(), locNum, rmtNum);

    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        Hccl::ExchangeUbBufferDto dto;
        dto.Deserialize(binaryStream);
        if (bufferVec.size() > pos) {
            continue;
        }
        HCCL_INFO("unpack %s pos=%u, dto %s", type.Describe().c_str(), pos, dto.Describe().c_str());
        if (dto.size == 0) {
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            bufferVec.push_back(nullptr);
        } else {
            bufferVec.push_back(std::make_unique<Hccl::RemoteUbRmaBuffer>(rdmaHandle_, dto));
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, bufferVec.back()->Describe().c_str());
        }
    }
}

void AicpuTsUboeUbgChannelHelper::RmtDrainBufferUnpackProc(Hccl::BinaryStream& binaryStream)
{
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] start unpack drain buffer", __func__);
    Hccl::ExchangeUbBufferDto dto;
    dto.Deserialize(binaryStream);

    if (dto.size == 0) {
        rmtDrainBuffer_ = nullptr;
        HCCL_WARNING("[AicpuTsUboeUbgChannelHelper::%s] unpack drain buffer dto is null", __func__);
    } else {
        rmtDrainBuffer_ = std::make_unique<Hccl::RemoteUbRmaBuffer>(rdmaHandle_, dto);
        HCCL_INFO(
            "[AicpuTsUboeUbgChannelHelper::%s] unpack drain buffer rmtDrainBuffer=%s", __func__,
            rmtDrainBuffer_->Describe().c_str());
    }
}

bool AicpuTsUboeUbgChannelHelper::ConnVecUnpackProc(Hccl::BinaryStream& binaryStream)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn connNum=%u, rmtConnNum=%u", connNum_, rmtConnNum);
    if (connNum_ != rmtConnNum) {
        MACRO_THROW(
            Hccl::InvalidParamsException,
            Hccl::StringFormat("connNum=%u is not equal to rmtConnNum=%u", connNum_, rmtConnNum));
    }

    bool result = false;
    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        Hccl::ExchangeUbConnDto rmtDto;
        rmtDto.Deserialize(binaryStream);
        HCCL_INFO("unpack connection pos=%u dto %s", pos, rmtDto.Describe().c_str());
        if (commonRes_.connVec[i]->GetStatus() != Hccl::RmaConnStatus::READY) {
            HCCL_INFO(
                "parse and import pos=%u, rmt dto to connection[%s]", pos, commonRes_.connVec[i]->Describe().c_str());
            commonRes_.connVec[i]->ParseRmtExchangeDto(rmtDto);
            commonRes_.connVec[i]->ImportRmtDto();
            result = true;
        }
    }
    return result;
}

static HcclResult SetUboeModuleDataName(Hccl::ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[SetModuleDataName] strcpy_s name %s failed", name.c_str());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

std::vector<char> AicpuTsUboeUbgChannelHelper::GetNotifyUniqueIds()
{
    HCCL_INFO("start packing all notify uniqueIds");
    std::vector<char> result(0);
    for (auto& it : commonRes_.notifyVec) {
        HCCL_INFO("AicpuTsUboeUbgChannelHelper Notify %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeUbgChannelHelper::GetSingleRmtBufferUniqueId(
    u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId) const
{
    Hccl::BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << tokenId;
    binaryStream << tokenValue;
    binaryStream << notifyId;
    HCCL_INFO("AicpuTsUboeUbgChannelHelper RmtBuffer[addr=0x%llx, size=0x%llx, notifyId=%u]", addr, size, notifyId);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char>
AicpuTsUboeUbgChannelHelper::GetRmtBufferUniqueIds(RemoteBufferVec& bufferVec, UboeRmtBufType type) const
{
    HCCL_INFO("start packing all remote buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto& it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(
                it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue(), it->GetNotifyId());
            HCCL_INFO("AicpuTsUboeUbgChannelHelper::GetRmtBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX);
            HCCL_INFO("AicpuTsUboeUbgChannelHelper::GetRmtBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char>
AicpuTsUboeUbgChannelHelper::GetLocBufferUniqueIds(LocalBufferVec& bufferVec, UboeRmtBufType type) const
{
    HCCL_INFO("start packing all local buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto& it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(
                it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue(), UINT32_MAX);
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX);
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeUbgChannelHelper::GetDrainUniqueIds() const
{
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] start get drain resource uniqueIds", __func__);
    std::vector<char> result(0);
    std::vector<char> uniqueId;

    // drain notify UniqueId
    if (drainNotify_ != nullptr) {
        auto dto = drainNotify_->GetExchangeDto();
        Hccl::ExchangeUbBufferDto* rawDto = static_cast<Hccl::ExchangeUbBufferDto*>(dto.get());
        uniqueId = GetSingleRmtBufferUniqueId(
            rawDto->addr, rawDto->size, rawDto->tokenId, rawDto->tokenValue, rawDto->notifyId);
        HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] %s", __func__, drainNotify_->Describe().c_str());
    } else {
        uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
        HCCL_WARNING("[AicpuTsUboeUbgChannelHelper::%s] drainNotify_ null buffer", __func__);
    }
    result.insert(result.end(), uniqueId.begin(), uniqueId.end());

    // drain buffer UniqueId
    if (rmtDrainBuffer_ != nullptr) {
        uniqueId = GetSingleRmtBufferUniqueId(
            rmtDrainBuffer_->GetAddr(), rmtDrainBuffer_->GetSize(), rmtDrainBuffer_->GetTokenId(),
            rmtDrainBuffer_->GetTokenValue(), rmtDrainBuffer_->GetNotifyId());
        HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] %s", __func__, rmtDrainBuffer_->Describe().c_str());
    } else {
        uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
        HCCL_WARNING("[AicpuTsUboeUbgChannelHelper::%s] rmtDrainBuffer_ null buffer", __func__);
    }
    result.insert(result.end(), uniqueId.begin(), uniqueId.end());

    return result;
}

std::vector<char> AicpuTsUboeUbgChannelHelper::GetConnUniqueIds()
{
    HCCL_INFO("start packing all conn uniqueIds");
    std::vector<char> result(0);
    for (auto& it : commonRes_.connVec) {
        HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] conn[%s]", __func__, it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeUbgChannelHelper::GetUniqueIdV2()
{
    if (channelStatus != ChannelStatus::READY) {
        MACRO_THROW(
            Hccl::InternalException,
            Hccl::StringFormat(
                "channel status[%d] is not ready[%d], please check.", channelStatus, ChannelStatus::READY));
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

    for (auto& it : commonRes_.bufferVec) {
        locBufferVec_.emplace_back(reinterpret_cast<Hccl::LocalUbRmaBuffer*>(it));
    }

    auto locBufferUniqueIds = GetLocBufferUniqueIds(locBufferVec_, UboeRmtBufType::BUFFER);
    binaryStream << locBufferUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds(rmtBufferVec_, UboeRmtBufType::BUFFER);
    binaryStream << rmtBufferUniqueIds;

    auto drainUniqueIds = GetDrainUniqueIds();
    binaryStream << drainUniqueIds;

    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

HcclResult AicpuTsUboeUbgChannelHelper::H2DResPack(std::vector<char>& buffer)
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetUboeModuleDataName(dataVec[resType], "AicpuTsUboeUbgChannelHelper"));

    std::vector<char> result;
    Hccl::BinaryStream binaryStream;
    binaryStream << GetUniqueIdV2();

    binaryStream.Dump(result);

    dataVec[resType].data = result;

    Hccl::AicpuResPackageHelper helper;
    buffer = helper.GetPackedData(dataVec);
    HCCL_INFO(
        "[AicpuTsUboeUbgChannelHelper][%s] Pack Buffer data[%p], Pack Buffer size[%zu].", __func__, buffer.data(),
        buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbgChannelHelper::Clean()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbgChannelHelper::Resume()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbgChannelHelper::NotifyRecord(const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbgChannelHelper::NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsUboeUbgChannelHelper::WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbgChannelHelper::Write(void* dst, const void* src, uint64_t len)
{
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbgChannelHelper::Read(void* dst, const void* src, uint64_t len)
{
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbgChannelHelper::ChannelFence()
{
    HCCL_INFO("[AicpuTsUboeUbgChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

} // namespace hcomm
