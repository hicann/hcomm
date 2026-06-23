/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "p2p_transport.h"

#include "ipc_local_notify.h"
#include "orion_adapter_rts.h"
#include "local_ipc_rma_buffer.h"
#include "exchange_ipc_notify_dto.h"
#include "exchange_ipc_buffer_dto.h"
#include "user_remote_mem_getter.h"

namespace Hccl {

P2PTransport::P2PTransport(CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData,
                           const Socket &socket)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::P2P)
{
    HcclResult result = FillTagVec(commonLocRes.bufferVec, localUserMemTag_);
    CHK_RET_THROW(InternalException,
        StringFormat("[P2PTransport][P2PTransport] failed to construct P2PTransport."),
        result);
}

P2PTransport::P2PTransport(CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData,
                           const Socket &socket, std::function<void(u32 streamId, u32 taskId, TaskParam taskParam)> callback)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::P2P, callback)
{
    HcclResult result = FillTagVec(commonLocRes.bufferVec, localUserMemTag_);
    CHK_RET_THROW(InternalException,
        StringFormat("[P2PTransport][P2PTransport] failed to construct P2PTransport."),
        result);
}

HcclResult P2PTransport::FillTagVec(std::vector<LocalRmaBuffer *> &bufferVec,
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> &tagVec)
{
    bufferNum += bufferVec.size();
    if (bufferNum == 0) {
        HCCL_WARNING("[P2PTransport][FillTagVec] bufferNum is 0.");
    }
    if (UNLIKELY(bufferNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR("[P2PTransport][FillTagVec] totalBufferNum[%u] exceeds limit[%u]", bufferNum, MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[P2PTransport][FillTagVec] bufferNum[%zu]", bufferVec.size());
    tagVec.reserve(bufferNum);
    uint32_t index = 0;
    for (auto &localRmaBuffer : bufferVec) {
        std::array<char, HCCL_RES_TAG_MAX_LEN> memTag{};
        if (localRmaBuffer == nullptr) {
            HCCL_WARNING("[P2PTransport][FillTagVec] localRmaBuffer is nullptr. memHandleNum[%u]", index);
        } else {
            CHK_PTR_NULL(localRmaBuffer->GetBuf());
            std::string tag = localRmaBuffer->GetBuf()->GetMemTag();
            if (UNLIKELY(tag.size() >= HCCL_RES_TAG_MAX_LEN)) {
                HCCL_ERROR("[P2PTransport][FillTagVec] tagSize exceeds limit[%u]", HCCL_RES_TAG_MAX_LEN);
                return HCCL_E_PARA;
            }
            CHK_SAFETY_FUNC_RET(memcpy_s(memTag.data(), memTag.size(), tag.c_str(), tag.size()));
            HCCL_INFO("[P2PTransport][FillTagVec] memHandleNum[%u] memTag[%s]", index, memTag.data());
        }
        tagVec.push_back(memTag);
        index++;
    }
    return HCCL_SUCCESS;
}

std::string P2PTransport::Describe() const
{
    string msg = StringFormat("P2PTransport:commonLocRes=%s, pidMsgSize=%u, myPid=%u, rmtPid=%u, rmtPidValid=%d,",
                              commonLocRes.Describe().c_str(), pidMsgSize, myPid, rmtPid, rmtPidValid);
    msg += StringFormat("exchangeDataSize=%u, ", exchangeDataSize);
    u32 pos = 0;
    for (auto &it : rmtNotifyVec) {
        msg += StringFormat("rmtNotify[%u]=%s, ", pos, it->Describe().c_str());
        pos++;
    }

    pos = 0;
    for (auto &it : rmtNotifyVec) {
        if (it != nullptr) {
            msg += StringFormat("rmtBuffer[%u]=%s, ", pos, it->Describe().c_str());
        } else {
            msg += StringFormat("rmtBuffer[%u]=nullptr, ", pos);
        }
        pos++;
    }
    return msg;
}

static void SubmitTask(const TaskP2pMemcpy &p2pMemcpy, const Stream &stream)
{
    HCCL_INFO("[P2PTransport::%s]not support, p2p dst addr[%llu], stream[%p]", __func__, p2pMemcpy.GetDstAddr(), stream.GetPtr());
}

static void SubmitTask(const TaskSdmaReduce &sdmaReduce, const Stream &stream)
{
    HCCL_INFO("[P2PTransport::%s]not support, sdmaReduce dst addr[%llu], stream[%p]", __func__, sdmaReduce.GetDstAddr(), stream.GetPtr());
}

template <typename TaskType> std::function<void(const BaseTask &, const Stream &)> GetSubmitP2PTaskFunction()
{
    return [](const BaseTask &task, const Stream &stream) {
        SubmitTask(static_cast<const TaskType &>(task), stream);
    };
}

std::map<TaskType, std::function<void(const BaseTask &, const Stream &)>> g_p2pTaskSubmitRuleMap
    = {{TaskType::P2P_MEMCPY, GetSubmitP2PTaskFunction<TaskP2pMemcpy>()},
       {TaskType::SDMA_REDUCE, GetSubmitP2PTaskFunction<TaskSdmaReduce>()}};

static void SubmitP2PTask(unique_ptr<BaseTask> task, const Stream &stream)
{
    if (task != nullptr) { // task为空的情况下，不需要提交task
        g_p2pTaskSubmitRuleMap.at(task->GetType())(*task.get(), stream);
    }
}

MemoryBuffer P2PTransport::GetLocMemBuffer(const RmaBufferSlice &locSlice) const
{
    return MemoryBuffer(locSlice.addr, locSlice.size, 0);
}

MemoryBuffer P2PTransport::GetRmtMemBuffer(const RmtRmaBufferSlice &rmtSlice) const
{
    return MemoryBuffer(rmtSlice.addr, rmtSlice.size, 0);
}

void P2PTransport::Post(u32 index, const Stream &stream)
{
    rmtNotifyVec[index]->Post(stream);
}

void P2PTransport::Read(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const Stream &stream)
{
    SqeConfig config;
    SubmitP2PTask(commonLocRes.connVec[0]->PrepareRead(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), config),
                  stream);
}

void P2PTransport::ReadReduce(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice,
                              const ReduceIn &reduceIn, const Stream &stream)
{
    SqeConfig config;
    SubmitP2PTask(commonLocRes.connVec[0]->PrepareReadReduce(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice),
                                                             reduceIn.dataType, reduceIn.reduceOp, config),
                  stream);
}

void P2PTransport::Write(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const Stream &stream)
{
    SqeConfig config;
    SubmitP2PTask(commonLocRes.connVec[0]->PrepareWrite(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), config),
                  stream);
}

void P2PTransport::WriteReduce(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice,
                               const ReduceIn &reduceIn, const Stream &stream)
{
    SqeConfig config;
    SubmitP2PTask(commonLocRes.connVec[0]->PrepareWriteReduce(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice),
                                                              reduceIn.dataType, reduceIn.reduceOp, config),
                  stream);
}

TransportStatus P2PTransport::GetStatus()
{
    if (baseStatus == TransportStatus::READY) {
        return baseStatus;
    }
    if (!IsSocketReady()) {
        p2pStatus = P2PStatus::INIT;
        return baseStatus;
    }
    baseStatus = TransportStatus::SOCKET_OK;

    switch (p2pStatus) {
        case P2PStatus::INIT:
            p2pStatus = P2PStatus::SOCKET_OK;
            break;
        case P2PStatus::SOCKET_OK:
            if (IsRmtPidValid()) { // rmt PID 有效，不需要交换PID，grant并发送data
                Grant();
                p2pStatus = P2PStatus::GRANT;
            } else { //  rmtPid无效，需要先交换PID
                SendPid();
                p2pStatus = P2PStatus::SEND_PID;
            }
            break;
        case P2PStatus::SEND_PID:
            RecvPid();
            rmtPidValid = true; // 收到 PID 了
            p2pStatus   = P2PStatus::RECV_PID;
            break;
        case P2PStatus::RECV_PID:
            Grant();
            p2pStatus = P2PStatus::GRANT;
            break;
        case P2PStatus::GRANT:
            SendExchangeData();
            p2pStatus = P2PStatus::SEND_DATA;
            break;
        case P2PStatus::SEND_DATA:
            RecvExchangeData();
            p2pStatus = P2PStatus::RECV_DATA;
            SetBaseStatusReady(); 
            break;
        default:
            break;
    }
    HCCL_INFO("%s, baseStatus=%s, p2pStatus = %s", GetLinkDescInfo().c_str(), baseStatus.Describe().c_str(),
               p2pStatus.Describe().c_str());
    return baseStatus;
}

bool P2PTransport::IsRmtPidValid() const
{
    // 优化方向：基于remoteAddr保存PID的单例，这样可以减少 PID交换
    return rmtPidValid;
}

void P2PTransport::SendPid()
{
    myPid = HrtDeviceGetBareTgid();
    HCCL_INFO("P2PTransport: send pid %u", myPid);

    BinaryStream binaryStream;
    binaryStream << myPid;

    std::vector<char> data;
    binaryStream.Dump(data);
    pidMsgSize = data.size();
    socket->Send(reinterpret_cast<u8 *>(&data[0]), data.size());

    HCCL_INFO("send pid %s, size=%llu, data=0x%s", GetLinkDescInfo().c_str(), data.size(),
               Bytes2hex(data.data(), data.size()).c_str());
}

void P2PTransport::RecvPid()
{
    std::vector<char> data(pidMsgSize);
    socket->Recv(reinterpret_cast<u8 *>(&data[0]), data.size());
    HCCL_INFO("recv pid %s, size=%llu, data=%s", GetLinkDescInfo().c_str(), data.size(),
               Bytes2hex(data.data(), data.size()).c_str());

    BinaryStream binaryStream(data);
    binaryStream >> rmtPid;
    HCCL_INFO("P2PTransport: recv pid %u", rmtPid);
}

void P2PTransport::Grant()
{
    // 暂时不做Grant处理
    return;
}

void P2PTransport::SendExchangeData()
{
    notifyNum = commonLocRes.notifyVec.size(); // 需要交换的notify数量
    bufferNum = commonLocRes.bufferVec.size(); // 需要交换的buffer数量

    HCCL_INFO("%s commLocResExchange %s, notifyNum=%u, bufferNum=%u", GetLinkDescInfo().c_str(),
               commonLocRes.Describe().c_str(), notifyNum, bufferNum);

    BinaryStream binaryStream;

    HandshakeMsgPack(binaryStream);
    NotifyVecPack(binaryStream);
    BufferVecPack(binaryStream);

    std::vector<char> data;
    binaryStream.Dump(data);
    exchangeDataSize = data.size();
    socket->Send(reinterpret_cast<u8 *>(&exchangeDataSize), sizeof(exchangeDataSize));
    socket->Send(reinterpret_cast<u8 *>(&data[0]), data.size());
    HCCL_INFO("send data %s, size=%llu, data=0x%s", GetLinkDescInfo().c_str(), data.size(),
               Bytes2hex(data.data(), data.size()).c_str());
}

void P2PTransport::RecvExchangeData()
{
    socket->Recv(reinterpret_cast<u8 *>(&exchangeDataSize), sizeof(exchangeDataSize));
    vector<char> data(exchangeDataSize);
    socket->Recv(reinterpret_cast<u8 *>(&data[0]), data.size());
    HCCL_INFO("recv data %s, size=%llu, data=%s", GetLinkDescInfo().c_str(), data.size(),
               Bytes2hex(data.data(), data.size()).c_str());

    BinaryStream binaryStream(data);
    HandshakeMsgUnpack(binaryStream);
    RmtNotifyVecUnpackProc(binaryStream);
    RmtBufferVecUnpackProc(binaryStream);

    HCCL_INFO("%s unpack success", GetLinkDescInfo().c_str());
}

void P2PTransport::RmtNotifyVecUnpackProc(BinaryStream &binaryStream)
{
    u32 rmtNotifyNum;
    binaryStream >> rmtNotifyNum;
    HCCL_INFO("unpack notify %s locNum=%u, rmtNum=%u", GetLinkDescInfo().c_str(), notifyNum, rmtNotifyNum);
    if (rmtNotifyNum != notifyNum) {
        MACRO_THROW(InvalidParamsException,
                    StringFormat("notifyNum=%u is not equal to rmtNotifyNum=%u", notifyNum, rmtNotifyNum));
    }

    rmtNotifyVec.clear(); // 清空remote资源
    for (u32 i = 0; i < rmtNotifyNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeIpcNotifyDto dto;
        dto.Deserialize(binaryStream);
        HCCL_INFO("unpack notify pos=%u dto %s", pos, dto.Describe().c_str());
        rmtNotifyVec.push_back(make_unique<IpcRemoteNotify>(dto));
        HCCL_INFO("unpack notify pos=%u, rmtNotify=%s", pos, rmtNotifyVec[i]->Describe().c_str());
    }
}

void P2PTransport::BufferVecPack(BinaryStream &binaryStream)
{
    binaryStream << bufferNum;
    HCCL_INFO("start pack %s bufferVec", transportType.Describe().c_str());
    u32 pos = 0;
    for (auto &it : commonLocRes.bufferVec) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u dto %s", pos, dto->Describe().c_str());
        } else { // 空的buffer，dto所有字段为0(size=0)
            ExchangeIpcBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u, dto is null %s", pos, exchangeDto.Describe().c_str());
        }
        pos++;
    }

    for (const auto& tag : localUserMemTag_) {
        // 逐个字节传输
        for (uint32_t i = 0; i < HCCL_RES_TAG_MAX_LEN; ++i) {
            binaryStream << static_cast<u8>(tag[i]);
        }
    }
}

void P2PTransport::RmtBufferVecUnpackProc(BinaryStream &binaryStream)
{
    u32 rmtBufferNum;
    binaryStream >> rmtBufferNum;
    HCCL_INFO("unpack buffer %s locNum=%u rmtNum=%u", GetLinkDescInfo().c_str(), bufferNum, rmtBufferNum);
    if (rmtBufferNum != bufferNum) {
        MACRO_THROW(InvalidParamsException,
                    StringFormat("bufferNum=%u is not equal to rmtBufferNum=%u", bufferNum, rmtBufferNum));
    }

    rmtBufferVec.clear();
    rmtRmaBufferVec.clear();
    for (u32 i = 0; i < rmtBufferNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeIpcBufferDto dto;
        dto.Deserialize(binaryStream);
        HCCL_INFO("unpack buffer pos=%u, dto %s", pos, dto.Describe().c_str());

        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            rmtBufferVec.push_back(nullptr);
            rmtRmaBufferVec.push_back((nullptr));
        } else { // size非0，则构造一个remote buffer
            rmtBufferVec.push_back(make_unique<RemoteIpcRmaBuffer>(dto));
            rmtRmaBufferVec.push_back(rmtBufferVec.back().get());
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, rmtBufferVec.back()->Describe().c_str());
        }
    }

    rmtMemTagTemp_.clear();
    rmtMemTagTemp_.resize(rmtBufferNum);
    for (auto& tag : rmtMemTagTemp_) {
        for (uint32_t i = 0; i < HCCL_RES_TAG_MAX_LEN; ++i) {
            u8 byte;
            binaryStream >> byte;
            tag[i] = static_cast<char>(byte);
        }
    }
    remoteUserMemTag_.insert(remoteUserMemTag_.end(), rmtMemTagTemp_.begin(), rmtMemTagTemp_.end());
}

std::vector<char> P2PTransport::GetUniqueId()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(InternalException, StringFormat("transport status is not ready, please check"));
    }
    u32          type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;

    // [header...][notifyUniqueId...][rmtNotifyUniqueId...][rmtBufferUniqueIds...]
    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtNotifyUniqueIds();
    binaryStream << rmtNotifyUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds();
    binaryStream << rmtBufferUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> P2PTransport::GetUniqueIdV2()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(InternalException, StringFormat("transport status is not ready, please check"));
    }
    u32          type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;

    // [header...][notifyUniqueId...][rmtNotifyUniqueId...][rmtBufferUniqueIds...]
    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtNotifyUniqueIds();
    binaryStream << rmtNotifyUniqueIds;

    auto locBufferUniqueIds = GetLocBufferUniqueIds();
    binaryStream << locBufferUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds();
    binaryStream << rmtBufferUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> P2PTransport::GetNotifyUniqueIds()
{
    HCCL_INFO("start packing all notify uniqueIds");
    std::vector<char> result(0);
    for (auto &it : commonLocRes.notifyVec) {
        HCCL_INFO("p2pMemTransport Notify %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> P2PTransport::GetSingleRmtNotifyUniqueId(u64 addr, u64 size, u32 notifyId) const
{
    BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << notifyId;
    HCCL_INFO("P2PTransport RmtNotifyAddr[addr=0x%llx, size=0x%llx, notifyId=%u]", addr, size, notifyId);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> P2PTransport::GetRmtNotifyUniqueIds() const
{
    HCCL_INFO("start packing all remote notify uniqueIds");
    std::vector<char> result(0);
    for (auto &it : rmtNotifyVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtNotifyUniqueId(it->GetAddr(), it->GetSize(), it->GetId());
            HCCL_INFO("P2PTransport::GetRmtNotifyUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtNotifyUniqueId(0, 0, 0); // 填充一个空的buffer
            HCCL_INFO("P2PTransport::GetRmtNotifyUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> P2PTransport::GetSingleBufferUniqueId(u64 addr, u64 size) const
{
    BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    HCCL_INFO("P2PTransport BufferAddr[addr=0x%llx, size=0x%llx]", addr, size);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> P2PTransport::GetLocBufferUniqueIds() const
{
    HCCL_INFO("start packing all local buffer uniqueIds");
    std::vector<char> result(0);
    for (auto &it : commonLocRes.bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleBufferUniqueId(it->GetAddr(), it->GetSize());
            HCCL_INFO("P2PTransport::GetLocBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleBufferUniqueId(0, 0); // 填充一个空的buffer
            HCCL_INFO("P2PTransport::GetLocBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> P2PTransport::GetRmtBufferUniqueIds() const
{
    HCCL_INFO("start packing all remote buffer uniqueIds");
    std::vector<char> result(0);
    for (auto &it : rmtBufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleBufferUniqueId(it->GetAddr(), it->GetSize());
            HCCL_INFO("P2PTransport::GetRmtBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleBufferUniqueId(0, 0); // 填充一个空的buffer
            HCCL_INFO("P2PTransport::GetRmtBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

HcclResult P2PTransport::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags) 
{
    CHK_PRT_RET(!remoteMem, HCCL_ERROR("[GetRemoteMem] remoteMem is nullptr"), HCCL_E_PARA);
    CHK_PRT_RET(!memNum, HCCL_ERROR("[GetRemoteMem] memNum is nullptr"), HCCL_E_PARA);
    HCCL_RUN_INFO("[P2PTransport]GetRemoteMem begin");
 
    *remoteMem = nullptr;
    *memNum = 0;
 
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
 
    uint32_t totalCount = rmtBufferVec.size();
    if (totalCount == 0) {
        HCCL_INFO("[GetRemoteMem] No remote memory regions available");
        return HCCL_SUCCESS;
    }
    // 释放之前的内存
    remoteMemsPtr_.reset();  
    remoteMemsPtr_ = std::make_unique<HcclMem[]>(totalCount);
    CHK_PTR_NULL(remoteMemsPtr_);

    for (uint32_t i = 0; i < totalCount; i++) {
        auto& rmtRmaBuffer = rmtBufferVec[i];
        remoteMemsPtr_[i].type = rmtRmaBuffer->GetMemType();
        remoteMemsPtr_[i].addr = reinterpret_cast<void *>(rmtRmaBuffer->GetAddr());
        remoteMemsPtr_[i].size = rmtRmaBuffer->GetSize();
        memTags[i] = const_cast<char*>(rmtRmaBuffer->GetMemTag().c_str());
        HCCL_INFO("[%s] addr[%p] size[%zu] rmtRmaBuffer[%p] memTags[%s]", 
            __func__, reinterpret_cast<void *>(rmtRmaBuffer->GetAddr()), rmtRmaBuffer->GetSize(), rmtRmaBuffer.get(), memTags[i]);
    }

    *memNum = totalCount;
    *remoteMem = remoteMemsPtr_.get();
    HCCL_INFO("[P2PTransport]GetRemoteMem end, memNum[%u]", totalCount);
    return HCCL_SUCCESS;
}

HcclResult P2PTransport::GetUserRemoteMem(CommMem **remoteMem, char ***memTags, uint32_t *memNum)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    if (rmtBufferVec.size() == 0) {
        HCCL_ERROR("[P2PTransport][GetUserRemoteMem] bufferNum is 0.");
        return HCCL_E_PARA;
    }
    uint32_t userMemCount = rmtBufferVec.size() - 1; // 默认 cclBuffer 数量为1，后续出现1的含义也是 cclBufferNum
    auto cacheBuilder = [](RemoteMemCtx<std::unique_ptr<RemoteIpcRmaBuffer>> &remoteMemCtx, uint32_t index) {
        auto &rmtBuffer = remoteMemCtx.rmtBufferVec[index + 1];
        if (rmtBuffer == nullptr) {
            return;
        }
        switch (rmtBuffer->GetMemType()) {
                case HCCL_MEM_TYPE_DEVICE:
                    remoteMemCtx.remoteUserMems[index].type = COMM_MEM_TYPE_DEVICE;
                    break;
                case HCCL_MEM_TYPE_HOST:
                    remoteMemCtx.remoteUserMems[index].type = COMM_MEM_TYPE_HOST;
                    break;
                default:
                    remoteMemCtx.remoteUserMems[index].type = COMM_MEM_TYPE_INVALID;
        }
        remoteMemCtx.remoteUserMems[index].addr = reinterpret_cast<void *>(rmtBuffer->GetAddr());
        remoteMemCtx.remoteUserMems[index].size = rmtBuffer->GetSize();
    };
    RemoteMemCtx<std::unique_ptr<RemoteIpcRmaBuffer>> remoteMemCtx{
        userMemCount, cacheValid_, rmtBufferVec, remoteUserMemTag_, remoteUserMems_, tagCopies_, tagPointers_,
        cacheBuilder, remoteMem, memTags, memNum};
    CHK_RET(GetRemoteUserMem(remoteMemCtx));
    return HCCL_SUCCESS;
}
} // namespace Hccl