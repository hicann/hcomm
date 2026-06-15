/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ub_mem_transport.h"
#include "serializable.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ub_conn_dto.h"
#include "ub_local_notify.h"
#include "local_ub_rma_buffer.h"
#include "sal.h"
#include "../../common/dlprof_func.h"
#include "user_remote_mem_getter.h"
#include "exception_util.h"
#include "env_config/env_config.h"

namespace Hccl {
constexpr u32    FINISH_MSG_SIZE             = 128;
constexpr char_t FINISH_MSG[FINISH_MSG_SIZE] = "Ub Comm Pipe ready!";
constexpr u32 ONE_MILLISECOND_OF_USLEEP      = 1000;

UbMemTransport::UbMemTransport(CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData,
                               const Socket &socket, RdmaHandle rdmaHandle1, LocCntNotifyRes &locCntNotifyRes1, 
                               bool isRecvFirst)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::UB), rdmaHandle(rdmaHandle1),
      locCntNotifyRes(locCntNotifyRes1), isRecvFirst_(isRecvFirst)
{
    HCCL_INFO("source: %s", locCntNotifyRes.Describe().c_str());
    HcclResult result = FillTagVec(commonLocRes.bufferVec, localUserMemTag_);
    CHK_RET_THROW(InternalException,
        StringFormat("[UbMemTransport][UbMemTransport] failed to construct UbMemTransport."),
        result);
}

UbMemTransport::UbMemTransport(CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData,
                               const Socket &socket, RdmaHandle rdmaHandle1, LocCntNotifyRes &locCntNotifyRes1, 
                               std::function<void(u32 streamId, u32 taskId, const TaskParam &taskParam)> callback)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::UB, callback), rdmaHandle(rdmaHandle1),
      locCntNotifyRes(locCntNotifyRes1)
{
    HCCL_INFO("source: %s", locCntNotifyRes.Describe().c_str());
    HcclResult result = FillTagVec(commonLocRes.bufferVec, localUserMemTag_);
    CHK_RET_THROW(InternalException,
        StringFormat("[UbMemTransport][UbMemTransport] failed to construct UbMemTransport."),
        result);
}

HcclResult UbMemTransport::FillTagVec(std::vector<LocalRmaBuffer *> &bufferVec,
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> &tagVec)
{
    bufferNum += bufferVec.size();
    if (bufferNum == 0) {
        HCCL_WARNING("[UbMemTransport][FillTagVec] bufferNum is 0.");
    }
    if (UNLIKELY(bufferNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR("[UbMemTransport][FillTagVec] totalBufferNum[%u] exceeds limit[%u]", bufferNum, MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[UbMemTransport][FillTagVec] bufferNum[%zu]", bufferVec.size());
    localUserMemTag_.reserve(bufferNum);
    uint32_t index = 0;
    for (auto &localRmaBuffer : bufferVec) {
        std::array<char, HCCL_RES_TAG_MAX_LEN> memTag{};
        if (localRmaBuffer == nullptr) {
            HCCL_WARNING("[UbMemTransport][FillTagVec] localRmaBuffer is nullptr. memHandleNum[%u]", index);
        } else {
            CHK_PTR_NULL(localRmaBuffer->GetBuf());
            std::string tag = localRmaBuffer->GetBuf()->GetMemTag();
            if (UNLIKELY(tag.size() >= HCCL_RES_TAG_MAX_LEN)) {
                HCCL_ERROR("[UbMemTransport][FillTagVec] tagSize exceeds limit[%u]", HCCL_RES_TAG_MAX_LEN);
                return HCCL_E_PARA;
            }
            CHK_SAFETY_FUNC_RET(memcpy_s(memTag.data(), memTag.size(), tag.c_str(), tag.size()));
            HCCL_INFO("[UbMemTransport][FillTagVec] memHandleNum[%u] memTag[%s]", index, memTag.data());
        }
        tagVec.push_back(memTag);
        index++;
    }
    return HCCL_SUCCESS;
}

std::string UbMemTransport::Describe() const
{
    string msg = StringFormat("UbMemTransport=[commonLocRes=%s, locCntNotifyRes=%s, ubStatus=%s, ",
                              commonLocRes.Describe().c_str(), locCntNotifyRes.Describe().c_str(),
                              ubStatus.Describe().c_str());
    msg += StringFormat("exchangeDataSize=%u, ", exchangeDataSize);
    msg += StringFormat("rmtNotifyNum=%zu, rmtCntNotifyVecNum=%zu]", rmtNotifyVec.size(), rmtCntNotifyVec.size());
    return msg;
}

HcclResult UbMemTransport::Describe(std::string &dfxMsg)
{
    HCCL_INFO("UbMemTransport Describe connNum[%u]", connNum);
    for (u32 i = 0; i < connNum; i++) {
        CHK_RET(commonLocRes.connVec[i]->Describe(dfxMsg));
    }
    return HCCL_SUCCESS;
}

MemoryBuffer UbMemTransport::GetLocMemBuffer(const RmaBufferSlice &locSlice) const
{
    return MemoryBuffer(locSlice.addr, locSlice.size, locSlice.buf->GetMemHandle());
}

MemoryBuffer UbMemTransport::GetRmtMemBuffer(const RmtRmaBufferSlice &rmtSlice) const
{
    return MemoryBuffer(rmtSlice.addr, rmtSlice.size, rmtSlice.buf->GetMemHandle());
}

MemoryBuffer UbMemTransport::GetRmtNotifyMemBuffer(u32 index)
{
    return MemoryBuffer(rmtNotifyVec[index]->GetAddr(), rmtNotifyVec[index]->GetSize(),
                        rmtNotifyVec[index]->GetMemHandle());
}

MemoryBuffer UbMemTransport::GetRmtCntNotifyMemBuffer(const WithNotifyIn &withNotify)
{
    auto index = withNotify.index_;
    return MemoryBuffer(rmtCntNotifyVec[index]->GetAddr(), rmtCntNotifyVec[index]->GetSize(),
                        rmtCntNotifyVec[index]->GetMemHandle());
}

static void SubmitTask(const TaskUbDbSend &ubSend, const Stream &stream)
{
    HCCL_INFO("SubmitTask UbDbSend ");
    HrtUbDbInfo info;
    info.dbNum = 1;
    info.wrCqe = 0; // 默认值是0 不会cqe  如果传1，驱动分发，会给hccl cqe，用于维护ci指针。
    info.info[0].functionId = ubSend.GetFuncId();
    info.info[0].dieId      = ubSend.GetDieId();
    info.info[0].jettyId    = ubSend.GetJettyId();
    info.info[0].piValue    = ubSend.GetPiVal();
    HrtUbDbSend(info, stream.GetPtr());
}

static void SubmitTask(const TaskUbDirectSend &ubDirectSend, const Stream &stream)
{
    HCCL_INFO("SubmitTask UbDirectSend");
    if (ubDirectSend.GetDwqeSize() != DWQE_SIZE_64 && ubDirectSend.GetDwqeSize() != DWQE_SIZE_128) {
        std::string msg
            = StringFormat("dwqe size is not valid, cannot submit task, dwqeSize=%u", ubDirectSend.GetDwqeSize());
        THROW<InternalException>(msg);
    }
    HrtUbWqeInfo info;
    info.wrCqe      = 0;
    info.functionId = ubDirectSend.GetFuncId();
    info.dieId      = ubDirectSend.GetDieId();
    info.jettyId    = ubDirectSend.GetJettyId();
    info.wqe        = const_cast<u8 *>(ubDirectSend.GetDwqePtr());
    info.wqePtrLen  = ubDirectSend.GetDwqeSize();
    info.wqeSize    = info.wqePtrLen == DWQE_SIZE_64 ? 0 : 1;
    HrtUbDirectSend(info, stream.GetPtr());
}

static void SubmitTask(const TaskWriteValue &taskWriteValue, const Stream &stream)
{
    HCCL_INFO("begin HrtWriteValue");
    HrtWriteValue(taskWriteValue.GetDbAddr(), taskWriteValue.GetPiVal(), stream.GetPtr());
    HCCL_INFO("finished HrtWriteValue");
}

template <typename TaskType> std::function<void(const BaseTask &, const Stream &)> GetSubmitUbTaskFunction()
{
    return [](const BaseTask &task, const Stream &stream) {
        SubmitTask(static_cast<const TaskType &>(task), stream);
    };
}

std::map<TaskType, std::function<void(const BaseTask &, const Stream &)>> g_ubTaskSubmitRuleMap
    = {{TaskType::UB_SEND, GetSubmitUbTaskFunction<TaskUbDbSend>()},
       {TaskType::UB_DIRECT_SEND, GetSubmitUbTaskFunction<TaskUbDirectSend>()},
       {TaskType::WRITE_VALUE, GetSubmitUbTaskFunction<TaskWriteValue>()}};

static void SubmitUbTask(unique_ptr<BaseTask> task, const Stream &stream)
{
    if (task != nullptr) {
        g_ubTaskSubmitRuleMap.at(task->GetType())(*task.get(), stream);
    }
}

void UbMemTransport::SubmitNotify(const MemoryBuffer &rmtNotify, u64 data, const Stream &stream)
{
    SqeConfig config;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareInlineWrite(rmtNotify, data, config), stream);
}

void UbMemTransport::Post(u32 index, const Stream &stream)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SubmitNotify(GetRmtNotifyMemBuffer(index), NORMAL_NOTIFY_VAL, stream);

    taskParam.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();;
    taskParam.taskPara.Notify.notifyID = rmtNotifyVec[index]->GetAddr();
    taskParam.taskPara.Notify.value = NORMAL_NOTIFY_VAL;
 
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::Wait(u32 index, const Stream &stream, u32 timeout)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
 
    commonLocRes.notifyVec[index]->Wait(stream, timeout);
 
    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = commonLocRes.notifyVec[index]->GetNotify()->GetId();
    taskParam.taskPara.Notify.value = NORMAL_NOTIFY_VAL;
 
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::Read(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const Stream &stream)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareRead(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), config),
                 stream);

    taskParam.taskType = TaskParamType::TASK_RDMA;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = reinterpret_cast<const void*>(locSlice.addr);
    taskParam.taskPara.DMA.dst = reinterpret_cast<const void*>(rmtSlice.addr);
    taskParam.taskPara.DMA.size = rmtSlice.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_READ;
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::ReadReduce(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice,
                                const ReduceIn &reduceIn, const Stream &stream)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareReadReduce(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice),
                                                            reduceIn.dataType, reduceIn.reduceOp, config),
                 stream);

    taskParam.taskType = TaskParamType::TASK_UB_REDUCE_INLINE;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = reinterpret_cast<const void*>(locSlice.addr);
    taskParam.taskPara.DMA.dst = reinterpret_cast<const void*>(rmtSlice.addr);
    taskParam.taskPara.DMA.size = rmtSlice.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_READ;
 
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::Write(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const Stream &stream)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareWrite(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), config),
                 stream);
    taskParam.taskType = TaskParamType::TASK_RDMA;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = reinterpret_cast<const void*>(locSlice.addr);
    taskParam.taskPara.DMA.dst = reinterpret_cast<const void*>(rmtSlice.addr);
    taskParam.taskPara.DMA.size = locSlice.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_WRITE;
 
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::WriteReduce(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice,
                                 const ReduceIn &reduceIn, const Stream &stream)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareWriteReduce(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice),
                                                             reduceIn.dataType, reduceIn.reduceOp, config),
                 stream);

    taskParam.taskType = TaskParamType::TASK_UB_REDUCE_INLINE;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = reinterpret_cast<const void*>(locSlice.addr);
    taskParam.taskPara.DMA.dst = reinterpret_cast<const void*>(rmtSlice.addr);
    taskParam.taskPara.DMA.size = locSlice.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_WRITE;
 
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::WriteWithNotify(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice,
                                     const WithNotifyIn &withNotify, const Stream &stream)
{
    if (locSlice.size == 0) {
        return SubmitWriteEmptyWithNotify(withNotify, stream);
    }

    if (withNotify.notifyType_ == TransportNotifyType::NORMAL) {
        return SubmitWriteWithNotify(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), NORMAL_NOTIFY_VAL,
                                     GetRmtNotifyMemBuffer(withNotify.index_), stream);
    } else if (withNotify.notifyType_ == TransportNotifyType::COUNT) {
        return SubmitWriteWithNotify(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), withNotify.userData_,
                                     GetRmtCntNotifyMemBuffer(withNotify), stream);
    } else {
        std::string msg = StringFormat("%s error", withNotify.Describe().c_str());
        THROW<InternalException>(msg);
    }
}

void UbMemTransport::WriteReduceWithNotify(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice,
                                           const ReduceIn &reduceIn, const WithNotifyIn &withNotify,
                                           const Stream &stream)
{
    if (locSlice.size == 0) {
        return SubmitWriteEmptyWithNotify(withNotify, stream);
    }

    if (withNotify.notifyType_ == TransportNotifyType::NORMAL) {
        SubmitWriteReduceWithNotify(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), reduceIn, NORMAL_NOTIFY_VAL,
                                    GetRmtNotifyMemBuffer(withNotify.index_), stream);
    } else if (withNotify.notifyType_ == TransportNotifyType::COUNT) {
        SubmitWriteReduceWithNotify(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), reduceIn, withNotify.userData_,
                                    GetRmtCntNotifyMemBuffer(withNotify), stream);
    } else {
        std::string msg = StringFormat("%s error", withNotify.Describe().c_str());
        THROW<InternalException>(msg);
    }
}

void UbMemTransport::SubmitWriteEmptyWithNotify(const WithNotifyIn &withNotify, const Stream &stream)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    u32 value = NORMAL_NOTIFY_VAL;

    if (withNotify.notifyType_ == TransportNotifyType::NORMAL) {
        SubmitNotify(GetRmtNotifyMemBuffer(withNotify.index_), NORMAL_NOTIFY_VAL, stream);
    } else if (withNotify.notifyType_ == TransportNotifyType::COUNT) {
        SubmitNotify(GetRmtCntNotifyMemBuffer(withNotify), withNotify.userData_, stream);
        value = withNotify.userData_;
    } else {
        std::string msg = StringFormat("%s error", withNotify.Describe().c_str());
        THROW<InternalException>(msg);
    }

    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.Notify.value = value;
 
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::SubmitWriteWithNotify(const MemoryBuffer &rmt, const MemoryBuffer &loc, u64 data,
                                           const MemoryBuffer &rmtNotify, const Stream &stream)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareWriteWithNotify(rmt, loc, data, rmtNotify, config), stream);

    taskParam.taskType = TaskParamType::TASK_WRITE_WITH_NOTIFY;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = reinterpret_cast<const void*>(loc.addr);
    taskParam.taskPara.DMA.dst = reinterpret_cast<const void*>(rmt.addr);
    taskParam.taskPara.DMA.size = loc.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_WRITE;
 
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::SubmitWriteReduceWithNotify(const MemoryBuffer &rmt, const MemoryBuffer &loc,
                                                 const ReduceIn &reduceIn, u64 data, const MemoryBuffer &rmtNotify,
                                                 const Stream &stream)
{
    TaskParam taskParam {};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareWriteReduceWithNotify(rmt, loc, reduceIn.dataType, reduceIn.reduceOp,
                                                                       data, rmtNotify, config),
                 stream);

    taskParam.taskType = TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = reinterpret_cast<const void*>(loc.addr);
    taskParam.taskPara.DMA.dst = reinterpret_cast<const void*>(rmt.addr);
    taskParam.taskPara.DMA.size = loc.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_WRITE;
 
    SaveDfxTaskInfo(taskParam);
}

bool UbMemTransport::IsResReady()
{
    for (auto &it : commonLocRes.connVec) {
        CHECK_NULLPTR(it,
            StringFormat("[UbMemTransport::%s] failed, connection pointer is nullptr", __func__));

        RmaConnType connType = it->GetRmaConnType();
        if (connType != RmaConnType::UB) {
            THROW<InternalException>("[UbMemTransport::%s] connection type[%s] is not ub",
                __func__, connType.Describe().c_str());
        }

        auto status = it->GetStatus();
        if (status != RmaConnStatus::EXCHANGEABLE &&
            status != RmaConnStatus::READY) {
            return false;
        }
    }

    HCCL_INFO("[UbMemTransport::IsResReady] all resources ready.");
    return true;
}

bool UbMemTransport::IsConnsReady()
{
    for (u32 i = 0; i < connNum; i++) {
        if (commonLocRes.connVec[i]->GetStatus() != RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("conns are ready.");
    return true;
}

TransportStatus UbMemTransport::GetStatus()
{
    if (baseStatus == TransportStatus::READY) {
        return baseStatus;
    } else if (baseStatus == TransportStatus::INIT) {
        ubStatus = UbStatus::INIT;
    }

    if (!IsSocketReady()) {
        return baseStatus;
    }

    switch (ubStatus) {
        case UbStatus::INIT:
            ubStatus = UbStatus::SEND_SIZE;
            baseStatus = TransportStatus::SOCKET_OK;
            break;
        case UbStatus::SEND_SIZE:
            if (IsResReady()) {
                SendDataSize();
                ubStatus = UbStatus::RECV_SIZE;
            }
            break;
        case UbStatus::RECV_SIZE:
            RecvDataSize();
            ubStatus = isRecvFirst_ ? UbStatus::RECV_DATA : UbStatus::SEND_DATA;
            break;
        case UbStatus::SEND_DATA:
            SendExchangeData();
            ubStatus = isRecvFirst_ ? UbStatus::PROCESS_DATA : UbStatus::RECV_DATA;
            break;
        case UbStatus::RECV_DATA:
            RecvExchangeData();
            ubStatus = isRecvFirst_ ? UbStatus::SEND_DATA : UbStatus::PROCESS_DATA;
            break;
        case UbStatus::PROCESS_DATA:
            if (RecvDataProcess()) { // 收消息中，如果设置到connection的建链，则需要发送 finish
                ubStatus = UbStatus::SEND_FIN;
            } else { // 不需要发送finish，则将transport状态调整为 ready
                SetBaseStatusReady();
                ubStatus = UbStatus::READY;
            }
            break;
        case UbStatus::SEND_FIN:
            if (IsConnsReady()) {
                SendFinish();
                ubStatus = UbStatus::RECV_FIN;
            }
            break;
        case UbStatus::RECV_FIN:
            RecvFinish();
            ubStatus = UbStatus::SET_READY;
            break;
        case UbStatus::SET_READY:
            SetBaseStatusReady();
            ubStatus = UbStatus::READY;
            break;
        default:
            break;
    }
    return baseStatus;
}

void UbMemTransport::SendDataSize()
{
    notifyNum    = commonLocRes.notifyVec.size(); // 需要交换的notify数量
    bufferNum    = commonLocRes.bufferVec.size(); // 需要交换的buffer数量
    connNum      = commonLocRes.connVec.size();
    cntNotifyNum = locCntNotifyRes.vec.size(); // 需要交换的cntNotify数量

    cntNotifyDescSize = locCntNotifyRes.desc.size(); // 需要交换的cntNotify数量

    HCCL_INFO("notifyNum=%u, bufferNum=%u, connNum=%u, cntNotifyNum=%u, cntNotifyDescSize=%u",
              notifyNum, bufferNum, connNum, cntNotifyNum, cntNotifyDescSize);

    BinaryStream binaryStream;
    HandshakeMsgPack(binaryStream);
    NotifyVecPack(binaryStream);
    BufferVecPack(binaryStream, commonLocRes.bufferVec, localUserMemTag_);
    CntNotifyVecPack(binaryStream);
    CntNotifyDescPack(binaryStream);
    ConnVecPack(binaryStream);

    binaryStream.Dump(sendData);
    u32 sendSize = sendData.size();

    // 发送数据包尺寸
    if (isHost_) {
        socket->Send(reinterpret_cast<u8 *>(&sendSize), sizeof(sendSize));
    } else {
        socket->SendAsync(reinterpret_cast<u8 *>(&sendSize), sizeof(sendSize));
    }
    HCCL_INFO("[UbMemTransport::%s] Send size[%u] of data success. [%zu] bytes sent.",
        __func__, sendSize, sizeof(sendSize));
}

void UbMemTransport::RecvDataSize()
{
    // 接收数据包尺寸
    if (isHost_) {
        socket->Recv(reinterpret_cast<u8 *>(&exchangeDataSize), sizeof(exchangeDataSize));
    } else {
        socket->RecvAsync(reinterpret_cast<u8 *>(&exchangeDataSize), sizeof(exchangeDataSize));
    }
    HCCL_INFO("[UbMemTransport::%s] Receive size[%u] of data success. [%zu] bytes received.",
        __func__, exchangeDataSize, sizeof(exchangeDataSize));
}

void UbMemTransport::SendExchangeData()
{
    if (isHost_) {
        socket->Send(reinterpret_cast<u8 *>(sendData.data()), sendData.size());
    } else {
        socket->SendAsync(reinterpret_cast<u8 *>(sendData.data()), sendData.size());
    }
    HCCL_INFO("send data %s, size=%llu", GetLinkDescInfo().c_str(), sendData.size());
}

void UbMemTransport::RecvExchangeData()
{
    recvData.resize(exchangeDataSize);
    if (isHost_) {
        socket->Recv(reinterpret_cast<u8 *>(recvData.data()), recvData.size());
    } else {
        socket->RecvAsync(reinterpret_cast<u8 *>(recvData.data()), recvData.size());
    }

    HCCL_INFO("recv data %s, size=%llu", GetLinkDescInfo().c_str(), recvData.size());
}

bool UbMemTransport::RecvDataProcess()
{
    HCCL_INFO("RecvDataProcess: link=%s, size=%llu, exchangeDataSize=%u", GetLinkDescInfo().c_str(), recvData.size(),
               exchangeDataSize);
    BinaryStream binaryStream(recvData);
    HandshakeMsgUnpack(binaryStream);
    RmtBufferVecUnpackProc(notifyNum, binaryStream, rmtNotifyVec, UbRmtBufType::NOTIFY);
    RmtBufferVecUnpackProc(bufferNum, binaryStream, rmtBufferVec, UbRmtBufType::BUFFER);
    RmtBufferVecUnpackProc(cntNotifyNum, binaryStream, rmtCntNotifyVec, UbRmtBufType::CNT_NOTIFY);
    CntNotifyDescUnpack(binaryStream);
    return ConnVecUnpackProc(binaryStream);
}

void UbMemTransport::BufferVecPack(BinaryStream &binaryStream, std::vector<LocalRmaBuffer *> &bufferVec,
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> &tagVec)
{
    binaryStream << static_cast<u32>(bufferVec.size());
    HCCL_INFO("start pack %s bufferVec", transportType.Describe().c_str());
    u32 pos = 0;
    for (auto &it : bufferVec) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u dto %s", pos, dto->Describe().c_str());
        } else { // 空的buffer，dto所有字段为0(size=0)
            ExchangeUbBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u, dto is null %s", pos, exchangeDto.Describe().c_str());
        }
        pos++;
    }

    for (const auto& tag : tagVec) {
        // 逐个字节传输
        for (uint32_t i = 0; i < HCCL_RES_TAG_MAX_LEN; ++i) {
            binaryStream << static_cast<u8>(tag[i]);
        }
    }
}

void UbMemTransport::CntNotifyVecPack(BinaryStream &binaryStream)
{
    binaryStream << cntNotifyNum;
    HCCL_INFO("pack UB cntNotify num=%u, %s", cntNotifyNum, GetLinkDescInfo().c_str());
    u32 pos = 0;
    for (auto &it : locCntNotifyRes.vec) {
        binaryStream << pos;
        std::unique_ptr<Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("pack cntNotify pos=%u, dto %s", pos, dto->Describe().c_str());
        pos++;
    }
}

void UbMemTransport::CntNotifyDescPack(BinaryStream &binaryStream)
{
    binaryStream << cntNotifyDescSize;
    HCCL_INFO("pack cntNotify desc size=%u %s", cntNotifyDescSize, GetLinkDescInfo().c_str());
    HCCL_INFO("pack cntNotify desc =%s", Bytes2hex(locCntNotifyRes.desc.data(), locCntNotifyRes.desc.size()).c_str());
    for (auto &it : locCntNotifyRes.desc) {
        binaryStream << it;
    }
}
void UbMemTransport::CntNotifyDescUnpack(BinaryStream &binaryStream)
{
    u32 descSize;
    binaryStream >> descSize;
    if (descSize != cntNotifyDescSize) {
        MACRO_THROW(InvalidParamsException,
                    StringFormat("CntNotifyDescUnpack size=%u is not equal to rmtNum=%u", descSize, cntNotifyDescSize));
    }
    rmtCntNotifyDesc.clear();
    u32 pos = 0;
    for (pos = 0; pos < descSize; pos++) {
        char c;
        binaryStream >> c;
        rmtCntNotifyDesc.push_back(c);
    }
    HCCL_INFO("unpack cntNotify Desc=%s", Bytes2hex(rmtCntNotifyDesc.data(), rmtCntNotifyDesc.size()).c_str());
}

void UbMemTransport::RmtBufferVecUnpackProc(u32 locNum, BinaryStream &binaryStream, RemoteBufferVec &bufferVec,
                                            UbRmtBufType type)
{
    u32 rmtNum;
    binaryStream >> rmtNum;
    if (UNLIKELY(type == UbRmtBufType::BUFFER && rmtNum > MAX_BUFFER_NUM)) {
        MACRO_THROW(InvalidParamsException,
            StringFormat("[UbMemTransport][RmtBufferVecUnpackProc] rmtNum[%u] exceeds limit[%u]",
            rmtNum, MAX_BUFFER_NUM));
    }

    // 允许本端和远端交换内存数量不一致
    HCCL_INFO("unpack %s %s, locNum=%u, rmtNum=%u", type.Describe().c_str(), GetLinkDescInfo().c_str(), locNum,
               rmtNum);

    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeUbBufferDto dto;
        dto.Deserialize(binaryStream);
        if (bufferVec.size() > pos) {
            // 对于之前已经加过的资源，无需追加
            continue;
        }

        HCCL_INFO("unpack %s pos=%u, dto %s", type.Describe().c_str(), pos, dto.Describe().c_str());
        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            bufferVec.push_back(nullptr);
            FillRmtRmaBufferVec(nullptr, type);
        } else { // size非0，则构造一个remote buffer
            bufferVec.push_back(make_unique<RemoteUbRmaBuffer>(rdmaHandle, dto));
            FillRmtRmaBufferVec(bufferVec.back().get(), type);
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, bufferVec.back()->Describe().c_str());
        }
    }

    rmtMemTagTemp_.clear();
    if (type == UbRmtBufType::BUFFER) {
        rmtMemTagTemp_.resize(rmtNum);
        for (auto& tag : rmtMemTagTemp_) {
            for (uint32_t i = 0; i < HCCL_RES_TAG_MAX_LEN; ++i) {
                u8 byte;
                binaryStream >> byte;
                tag[i] = static_cast<char>(byte);
            }
        }
    }
    remoteUserMemTag_.insert(remoteUserMemTag_.end(), rmtMemTagTemp_.begin(), rmtMemTagTemp_.end());
}

bool UbMemTransport::ConnVecUnpackProc(BinaryStream &binaryStream)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn %s connNum=%u, rmtConnNum=%u", GetLinkDescInfo().c_str(), connNum, rmtConnNum);
    if (connNum != rmtConnNum) {
        MACRO_THROW(InvalidParamsException,
                    StringFormat("connNum=%u is not equal to rmtConnNum=%u", connNum, rmtConnNum));
    }

    bool result = false; // 不需要发送 finish
    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeUbConnDto rmtDto;
        rmtDto.Deserialize(binaryStream);
        HCCL_INFO("unpack connection pos=%u dto %s", pos, rmtDto.Describe().c_str());
        if (commonLocRes.connVec[i]->GetStatus() != RmaConnStatus::READY) {
            HCCL_INFO("parse and import pos=%u, rmt dto to connection[%s]", pos,
                       commonLocRes.connVec[i]->Describe().c_str());
            commonLocRes.connVec[i]->ParseRmtExchangeDto(rmtDto);
            commonLocRes.connVec[i]->ImportRmtDto();
            result = true; // connection 建链，需要发送finish
        }
    }
    return result;
}

void UbMemTransport::FillRmtRmaBufferVec(RemoteRmaBuffer *rmaBuffer, UbRmtBufType type)
{
    if (type == UbRmtBufType::BUFFER) {
        rmtRmaBufferVec.push_back(rmaBuffer);
    }
}

void UbMemTransport::SendFinish()
{
    HCCL_INFO("start send Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
    sendFinishMsg = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    if (isHost_) {
        socket->Send(reinterpret_cast<u8 *>(sendFinishMsg.data()), FINISH_MSG_SIZE);
    } else {
        socket->SendAsync(reinterpret_cast<u8 *>(sendFinishMsg.data()), FINISH_MSG_SIZE);
    }
    HCCL_INFO("end send Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
}

void UbMemTransport::RecvFinish()
{
    recvFinishMsg.resize(FINISH_MSG_SIZE);
    HCCL_INFO("start recv Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
    if (isHost_) {
        socket->Recv(reinterpret_cast<u8 *>(recvFinishMsg.data()), FINISH_MSG_SIZE);
    } else {
        socket->RecvAsync(reinterpret_cast<u8 *>(recvFinishMsg.data()), FINISH_MSG_SIZE);
    }
    HCCL_INFO("end recv Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
}

std::vector<char> UbMemTransport::GetUniqueId()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(InternalException, StringFormat("transport status is not ready, please check"));
    }
    u32          type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << static_cast<u32>(rmtBufferVec.size());
    binaryStream << connNum;

    // [header...][notifyUniqueId...][rmtNotifyUniqueId...][rmtBufferUniqueIds...]
    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtBufferUniqueIds(rmtNotifyVec, UbRmtBufType::NOTIFY);
    binaryStream << rmtNotifyUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds(rmtBufferVec, UbRmtBufType::BUFFER);
    binaryStream << rmtBufferUniqueIds;

    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> UbMemTransport::GetUniqueIdV2()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(InternalException, StringFormat("transport status[%d] is not ready[%d], please check.",
            baseStatus, TransportStatus::READY));
    }
    u32          type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << static_cast<u32>(rmtBufferVec.size());
    binaryStream << connNum;
 
    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;
 
    auto rmtNotifyUniqueIds = GetRmtBufferUniqueIds(rmtNotifyVec, UbRmtBufType::NOTIFY);
    binaryStream << rmtNotifyUniqueIds;
 
    for (auto &it : commonLocRes.bufferVec) {
        locBufferVec.emplace_back(reinterpret_cast<LocalUbRmaBuffer *>(it));
    }
 
    auto locBufferUniqueIds = GetLocBufferUniqueIds(locBufferVec, UbRmtBufType::BUFFER);
    binaryStream << locBufferUniqueIds;
 
    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds(rmtBufferVec, UbRmtBufType::BUFFER);
    binaryStream << rmtBufferUniqueIds;
 
    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;
 
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> UbMemTransport::PackConnData()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(InternalException, StringFormat("transport status[%d] is not ready[%d], please check.",
            baseStatus, TransportStatus::READY));
    }
    u32          type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << connNum;
 
    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;
 
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> UbMemTransport::GetSingleRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue) const
{
    BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << tokenId;
    binaryStream << tokenValue;
    HCCL_INFO("UbMemTransport RmtBuffer[addr=0x%llx, size=0x%llx]", addr, size);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> UbMemTransport::GetNotifyUniqueIds()
{
    HCCL_INFO("start packing all notify uniqueIds");
    std::vector<char> result(0);
    for (auto &it : commonLocRes.notifyVec) {
        HCCL_INFO("ubMemTransport Notify %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> UbMemTransport::GetRmtBufferUniqueIds(RemoteBufferVec &bufferVec, UbRmtBufType type) const
{
    HCCL_INFO("start packing all remote buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto &it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue());
            HCCL_INFO("UbMemTransport::GetRmtBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0); // 填充一个空的buffer
            HCCL_INFO("UbMemTransport::GetRmtBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> UbMemTransport::GetLocBufferUniqueIds(LocalBufferVec &bufferVec, UbRmtBufType type) const
{
    HCCL_INFO("start packing all local buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto &it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue());
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0); // 填充一个空的buffer
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> UbMemTransport::GetConnUniqueIds()
{
    HCCL_INFO("start packing all conn uniqueIds");
    std::vector<char> result(0);
    for (auto &it : commonLocRes.connVec) {
        HCCL_INFO("[UbMemTransport::%s] conn[%s]", __func__, it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

void UbMemTransport::SaveDfxTaskInfo(const TaskParam &taskParam)
{
    u32 taskId;
    u32 streamId;
    HrtGetTaskIdAndStreamID(taskId, streamId);

    callback(streamId, taskId, taskParam);
}

HcclResult UbMemTransport::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags) 
{
    CHK_PRT_RET(!remoteMem, HCCL_ERROR("[GetRemoteMem] remoteMem is nullptr"), HCCL_E_PARA);
    CHK_PRT_RET(!memNum, HCCL_ERROR("[GetRemoteMem] memNum is nullptr"), HCCL_E_PARA);
 
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
        HCCL_INFO("[%s] addr[%p] size[%zu] rmtRmaBuffer[%p]", 
            __func__, reinterpret_cast<void *>(rmtRmaBuffer->GetAddr()), rmtRmaBuffer->GetSize(), rmtRmaBuffer.get());
    }

    *memNum = totalCount;
    *remoteMem = remoteMemsPtr_.get();
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::GetUserRemoteMem(CommMem **remoteMem, char ***memTags, uint32_t *memNum)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    if (rmtBufferVec.size() == 0) {
        HCCL_ERROR("[UbMemTransport][GetUserRemoteMem] bufferNum is 0.");
        return HCCL_E_PARA;
    }
    uint32_t userMemCount = rmtBufferVec.size() - 1; // 默认 cclBuffer 数量为1，后续出现1的含义也是 cclBufferNum
    auto cacheBuilder = [](RemoteMemCtx<std::unique_ptr<RemoteUbRmaBuffer>> &remoteMemCtx, uint32_t index) {
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
    RemoteMemCtx<std::unique_ptr<RemoteUbRmaBuffer>> remoteMemCtx{
        userMemCount, cacheValid_, rmtBufferVec, remoteUserMemTag_, remoteUserMems_, tagCopies_, tagPointers_,
        cacheBuilder, remoteMem, memTags, memNum};
    CHK_RET(GetRemoteUserMem(remoteMemCtx));
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::CheckSocketStatus(std::string socketOpreator)
{
    CHK_PTR_NULL(socket);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while(true) {
        SocketStatus socketStatus = socket->GetAsyncStatus();
        if (socketStatus == SocketStatus::OK) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            HCCL_INFO("[UbMemTransport][%s] socket transport operation[%s] success, elapsed[%lld]ms, retryCount[%u]",
                __func__, socketOpreator.c_str(), elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout ||
            socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            HCCL_ERROR("[UbMemTransport][%s] socket transport operation[%s] timeout after %lld sec, elapsed[%lld]ms, retryCount[%u]",
                __func__, socketOpreator.c_str(), timeout, elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::UpdateMemInfo(std::vector<LocalRmaBuffer *> &bufferVecTemp)
{
    if (bufferVecTemp.size() == 0) {
        HCCL_WARNING("[UbMemTransport][UpdateMemInfo] bufferNum is 0.");
        return HCCL_SUCCESS;
    }
    locMemTagTemp_.clear();
    CHK_RET(FillTagVec(bufferVecTemp, locMemTagTemp_));
    HCCL_INFO("[UbMemTransport][UpdateMemInfo] bufferNum[%zu]", bufferVecTemp.size());
    sendData.clear();
    BinaryStream sendStream;
    std::vector<std::unique_ptr<RemoteUbRmaBuffer>> rmtBufferTemp{};
    TRY_CATCH_RETURN(
        [&]() -> void {
            BufferVecPack(sendStream, bufferVecTemp, locMemTagTemp_);
            sendStream.Dump(sendData);
            u32 sendSize = sendData.size();
            socket->SendAsync(reinterpret_cast<u8 *>(&sendSize), sizeof(sendSize));
            HCCL_INFO("[UbMemTransport][UpdateMemInfo] Send size[%u] of data success. [%zu] bytes sent.",
                sendSize, sizeof(sendSize));
            HcclResult result = CheckSocketStatus("SendDataSize");
            CHK_RET_THROW(InternalException,
                StringFormat("[UbMemTransport][UpdateMemInfo] failed to send dataSize."),
                result);
            RecvDataSize();
            result = CheckSocketStatus("RecvDataSize");
            CHK_RET_THROW(InternalException,
                StringFormat("[UbMemTransport][UpdateMemInfo] failed to receive dataSize."),
                result);
            SendExchangeData();
            result = CheckSocketStatus("SendExchangeData");
            CHK_RET_THROW(InternalException,
                StringFormat("[UbMemTransport][UpdateMemInfo] failed to send data."),
                result);
            RecvExchangeData();
            result = CheckSocketStatus("RecvExchangeData");
            CHK_RET_THROW(InternalException,
                StringFormat("[UbMemTransport][UpdateMemInfo] failed to receive data."),
                result);
            BinaryStream recvStream(recvData);
            RmtBufferVecUnpackProc(bufferNum, recvStream, rmtBufferTemp, UbRmtBufType::BUFFER);
        }());
    rmtBufferVec.insert(rmtBufferVec.end(), std::make_move_iterator(rmtBufferTemp.begin()),
        std::make_move_iterator(rmtBufferTemp.end()));
    commonLocRes.bufferVec.insert(commonLocRes.bufferVec.end(), bufferVecTemp.begin(), bufferVecTemp.end());
    localUserMemTag_.insert(localUserMemTag_.end(), locMemTagTemp_.begin(), locMemTagTemp_.end());
    cacheValid_ = false;
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::Init() 
{
    for (auto& ubConn : commonLocRes.connVec) {
        TRY_CATCH_RETURN(ubConn->Connect());
    }
    return HCCL_SUCCESS;
}
 
HcclResult UbMemTransport::DeInit() const
{
    socket->Destroy();
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::GetRemoteSeg(const void* addr, u64 len, u64 *seg)
{
    if (rmtBufferVec.empty()) {
        HCCL_ERROR("[UbMemTransport::%s] rmtBufferVec is empty.", __func__);
        return HCCL_E_INTERNAL;
    }

    bool isAddrInRange = false;
    for (auto &it : rmtBufferVec) {
        Buffer iterBuf(it->GetAddr(), it->GetSize());
        if (iterBuf.Contains(reinterpret_cast<uintptr_t>(addr), len)) {
            *seg = it->GetSegVa();
            isAddrInRange = true;
            break;
        }
    }

    if (!isAddrInRange) {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl