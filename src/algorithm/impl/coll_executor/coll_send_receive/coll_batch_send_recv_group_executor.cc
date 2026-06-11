/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_batch_send_recv_group_executor.h"

namespace hccl {
constexpr u64 BIG_DATA = 128 * 1024;

CollBatchSendRecvGroupExecutor::CollBatchSendRecvGroupExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBatchSendRecvExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollBatchSendRecvGroupExecutor::CalcBufferSliceSize()
{
    u32 bufferSliceNum = GROUP_MAX_CONCURRENT;
    u32 alignSize = HCCL_MIN_SLICE_ALIGN_910B; // 对齐
    bufferSliceSize_ = algResResp_->cclInputMem.size() / alignSize / bufferSliceNum * alignSize; // cclInp
    HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcBufferSliceSize] bufferSliceSize_[%llu]", bufferSliceSize_);
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::OrganizeSendItemByStream()
{
    sendQueueBySendstream_.resize(sendStreamNum_);
    HCCL_INFO("[OrganizeSendItemByStream] sendStreamNum_[%u]", sendStreamNum_);
    while (!sendDeque_.empty()) {
        HcclSendRecvItem* curr = sendDeque_.front();
        CHK_PTR_NULL(curr);
        sendQueueBySendstream_[curr->remoteRank % sendStreamNum_].push_back(curr);
        sendDeque_.pop_front();
    }
    HCCL_INFO("OrganizeSendItemByStream Done!");
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::OrganizeRecvItemByStream()
{
    recvQueueByRecvstream_.resize(recvStreamNum_);
    HCCL_INFO("[OrganizeRecvItemByStream] recvStreamNum_[%u]", recvStreamNum_);
    while (!recvDeque_.empty()) {
        HcclSendRecvItem* curr = recvDeque_.front();
        CHK_PTR_NULL(curr);
        recvQueueByRecvstream_[curr->remoteRank % recvStreamNum_].push_back(curr);
        recvDeque_.pop_front();
    }
    HCCL_INFO("OrganizeRecvItemByStream Done!");
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::isGroupBigCount(HcclSendRecvItem *sendRecvInfo, u32 itemNum, bool& isBig) const
{
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        CHK_PTR_NULL(sendRecvInfo->buf);
        u32 unitSize = SIZE_TABLE[sendRecvInfo->dataType];
        if (sendRecvInfo->count * unitSize > BIG_DATA) { // 只要有一个big，就认为是big
            isBig = true;
            break;
        }
        sendRecvInfo++;
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBatchSendRecvGroupExecutor] groupsendrecv starts.");
    
    sendStreamNum_ = GROUP_MAX_CONCURRENT;
    recvStreamNum_ = GROUP_MAX_CONCURRENT;
    HCCL_INFO("[Orchestrate] sendStreamNum_[%u], recvStreamNum_[%u]", sendStreamNum_, recvStreamNum_);

    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));
    CHK_RET(GetPairWiseList(param.BatchSendRecvDataDes.sendRecvItemsPtr, param.BatchSendRecvDataDes.itemNum));
    CHK_RET(ProcessSelfSendRecvTasks(param.stream));
    if (topoAttr_.userRankSize == 1) {
        HCCL_INFO("tag[%s] BatchSendRecvGroup Excutor orchestrate success, take time [%lld]us.",
            param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
        return HCCL_SUCCESS;
    }
    CHK_RET(CalcBufferSliceSize());

    bool isBig = false;
    CHK_RET(isGroupBigCount(param.BatchSendRecvDataDes.sendRecvItemsPtr, param.BatchSendRecvDataDes.itemNum, isBig));
    if (isBig) {
        CHK_RET(OrganizeSendItemByStream());
        CHK_RET(OrganizeRecvItemByStream());

        CHK_RET(CalcSendSlices());
        CHK_RET(CalcRecvSlices());

        CHK_RET(RunLoopBig(param));
    } else {
        // 不用按照streamId入队
        CHK_RET(CalcSendSlicesSmall());
        CHK_RET(CalcRecvSlicesSmall());
        CHK_RET(RunLoopSmall(param));
    }

    HCCL_INFO("tag[%s] BatchSendRecvGroup Excutor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::MainPostSubWait(Stream& mainStream)
{
    // 主流通知所有从流开始
    for (u32 i  = 0; i < sendStreamNum_; i++){
        if (!sendDataSilcesBySendStream_[i].empty()){ // 只对有任务的流进行同步，减少开销
            CHK_RET(LocalNotify::Post(mainStream, dispatcher_, algResResp_->notifiesAux[i], PROF_STAGE_0));
            CHK_RET(LocalNotify::Wait(algResResp_->slaveStreams[i], dispatcher_, algResResp_->notifiesAux[i], PROF_STAGE_0));
            HCCL_INFO("MainPost, Send[%u] Wait", i);
        }
    }

    for (u32 i  = 0; i < recvStreamNum_; i++){
        if (!recvDataSilcesByRecvStream_[i].empty()){
            CHK_RET(LocalNotify::Post(mainStream, dispatcher_, algResResp_->notifiesAux[i + sendStreamNum_], PROF_STAGE_0));
            CHK_RET(LocalNotify::Wait(algResResp_->slaveStreams[i + sendStreamNum_], dispatcher_, algResResp_->notifiesAux[i + sendStreamNum_], PROF_STAGE_0));
            HCCL_INFO("MainPost, Recv[%u] Wait", i);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::MainWaitSubPost(Stream& mainStream)
{
    // 最后主流等待所有从流结束
    for (u32 i  = 0; i < sendStreamNum_; i++){
        if (!sendDataSilcesBySendStream_[i].empty()){ // 只对有任务的流进行同步，减少开销
            CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[i], dispatcher_, algResResp_->notifiesMain[i], PROF_STAGE_0));
            CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, algResResp_->notifiesMain[i], PROF_STAGE_0));
            HCCL_INFO("MainWait, Send[%u] Post", i);
        }
    }

    for (u32 i  = 0; i < recvStreamNum_; i++){
        if (!recvDataSilcesByRecvStream_[i].empty()){
            CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[i + sendStreamNum_], dispatcher_, algResResp_->notifiesMain[i + sendStreamNum_], PROF_STAGE_0));
            CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, algResResp_->notifiesMain[i + sendStreamNum_], PROF_STAGE_0));
            HCCL_INFO("MainWait, Recv[%u] Post", i);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CopyLocalDataForARound(Stream& mainStream)
{
    for (u32 i = 0; i < sendStreamNum_; i++){
        if (sendDataSilcesBySendStream_[i].empty()) {
            continue;
        }
        SendRecvSlice& slice = sendDataSilcesBySendStream_[i].front();
        DeviceMem inMem(slice.addr, slice.size);
        u64 offset = bufferSliceSize_ * (slice.remoteRank % sendStreamNum_);
        DeviceMem inCommMem = algResResp_->cclInputMem.range(offset, slice.size);
        HCCL_INFO("[CopyLocalDataForARound] inMem ptr[%p], size[%llu]", inMem.ptr(), inMem.size());
        HCCL_INFO("[CopyLocalDataForARound] inCommMem ptr[%p], size[%llu]", inCommMem.ptr(), inCommMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, mainStream));
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::RunTasksBig(OpParam& param)
{
    u32 nonEmptySendStream = 0;
    u32 nonEmptyRecvStream = 0;
    for (u32 i = 0; i < sendStreamNum_; i++) {
        if (!sendDataSilcesBySendStream_[i].empty()) nonEmptySendStream++;
    }
    for (u32 i = 0; i < recvStreamNum_; i++) {
        if (!recvDataSilcesByRecvStream_[i].empty()) nonEmptyRecvStream++;
    }

    while (nonEmptySendStream > 0 || nonEmptyRecvStream > 0) {
        CHK_RET(CopyLocalDataForARound(param.stream));
        HCCL_INFO("nonEmptySendStream[%u], nonEmptyRecvStream[%u]", nonEmptySendStream, nonEmptyRecvStream);
        CHK_RET(MainPostSubWait(param.stream));

        for (u32 i = 0; i < sendStreamNum_; i++){
            if (!sendDataSilcesBySendStream_[i].empty()) {
                CHK_RET(ProcessSendStreamDataSlice(algResResp_->slaveStreams[i], false, false, true, i));
            }
        }
        for (u32 i = 0; i < recvStreamNum_; i++){
            if (!recvDataSilcesByRecvStream_[i].empty()) {
                CHK_RET(ProcessRecvStreamDataSlice(algResResp_->slaveStreams[i + sendStreamNum_], false, false, true, i));
            }
        }
        CHK_RET(MainWaitSubPost(param.stream));

        // 更新DataSlices
        for (u32 i = 0; i < sendStreamNum_; i++){
            if (!sendDataSilcesBySendStream_[i].empty()) {
                sendDataSilcesBySendStream_[i].pop_front();
                if (sendDataSilcesBySendStream_[i].empty()) {
                    --nonEmptySendStream;
                    HCCL_INFO("[RunLoopBig] nonEmptySendStream[%u]", nonEmptySendStream);
                }
            }
        }
        for (u32 i = 0; i < recvStreamNum_; i++){
            if (!recvDataSilcesByRecvStream_[i].empty()) {
                recvDataSilcesByRecvStream_[i].pop_front();
                if (recvDataSilcesByRecvStream_[i].empty()) {
                    --nonEmptyRecvStream;
                    HCCL_INFO("[RunLoopBig] nonEmptyRecvStream[%u]", nonEmptyRecvStream);
                }
            }
        }
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::RunLoopBig(OpParam& param)
{
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        auto meta = HcclOpMetaInfo::GetOneForBatchSendRecv();
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream,
            dispatcher_));
    }
    bool isSetNormalMode = false; // 设置过一次就不需要再设置了
    for (u32 i = 0; i < sendDataSilces_.size(); ++i) {
        SendRecvSlice& slice = sendDataSilces_[i];
        LINK targetLink;
        CHK_RET(GetSendTargetLink(slice.remoteRank, targetLink));
        if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
            CHK_RET(SetNormalMode(dispatcher_));
            isSetNormalMode = true;
            HCCL_INFO("[CollBatchSendRecvGroupExecutor][RunLoopBig]Send Set NormalMode dispatcher");
            break;
        }
    }

    for (u32 i = 0; i < recvDataSilces_.size() && !isSetNormalMode; ++i) {
        SendRecvSlice& slice = recvDataSilces_[i];
        LINK targetLink;
        CHK_RET(GetRecvTargetLink(slice.remoteRank, targetLink));
        if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
            CHK_RET(SetNormalMode(dispatcher_));
            HCCL_INFO("[CollBatchSendRecvGroupExecutor][RunLoopBig]Recv Set NormalMode dispatcher");
            break;
        }
    }
    CHK_RET(RunTasksBig(param));
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream, dispatcher_));
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
        HCCL_INFO("LaunchTaskExtend!");
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::RunLoopSmall(OpParam& param)
{
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        auto meta = HcclOpMetaInfo::GetOneForBatchSendRecv();
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream,
            dispatcher_));
    }
    bool isSetNormalMode = false; // 设置过一次就不需要再设置了
    for (u32 i = 0; i < sendDataSilces_.size(); ++i) {
        SendRecvSlice& slice = sendDataSilces_[i];
        LINK targetLink;
        CHK_RET(GetSendTargetLink(slice.remoteRank, targetLink));
        if (TransportType::TRANS_TYPE_DEVICE_DIRECT == targetLink->GetTransportType()) {
            if (!isSetNormalMode) {
                CHK_RET(SetNormalMode(dispatcher_));
                isSetNormalMode = true;
            }
            HCCL_INFO("[CollBatchSendRecvGroupExecutor][RunLoopSmall]Send Set NormalMode true");
            break;
        }
    }

    for (u32 i = 0; i < recvDataSilces_.size() && !isSetNormalMode; ++i) {
        SendRecvSlice& slice = recvDataSilces_[i];
        LINK targetLink;
        CHK_RET(GetRecvTargetLink(slice.remoteRank, targetLink));
        if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
            CHK_RET(SetNormalMode(dispatcher_));
            HCCL_INFO("[CollBatchSendRecvGroupExecutor][RunLoopSmall]Recv Set NormalMode dispatcher");
            break;
        }
    }

    CHK_RET(ProcessDataSliceSmall(param));

    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(algResResp_->cclInputMem,
            algResResp_->cclOutputMem, param.stream, dispatcher_));
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
        HCCL_INFO("LaunchTaskExtend!");
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcSendSlices()
{
    sendDataSilcesBySendStream_.resize(sendStreamNum_);
    for (u32 i = 0; i < sendStreamNum_; i++) { // 一个个stream处理
        auto sendQueueInner = sendQueueBySendstream_[i]; 
        // 遍历这个stream对应的queue中的所有item，每次取一个slice，然后再重新遍历，直到所有的itemcount减为0
        u32 emptyItem = 0;
        while (emptyItem < sendQueueInner.size()) {
            for (u32 j = 0; j < sendQueueInner.size(); j++){
                HcclSendRecvItem* sendRecvItem = sendQueueInner[j];
                if (sendRecvItem->count == 0) {
                    continue;
                }
                u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
                u64 maxCountPerLoop = CalcSendLoopMaxCount(unitSize);
                u8 *curInputPtr = static_cast<u8 *>(sendRecvItem->buf);
                CHK_PTR_NULL(curInputPtr);
                u64 curCount = (sendRecvItem->count > maxCountPerLoop) ? maxCountPerLoop : sendRecvItem->count;
                u64 curSize = curCount * unitSize; // 单位：字节
                sendDataSilcesBySendStream_[i].emplace_back(curInputPtr, curSize, sendRecvItem->remoteRank);
                sendRecvItem->count -= curCount; // 更新剩余的count, 若sendRecvItem->count <= maxCountPerLoop，sendRecvItem->count会更新为0
                sendRecvItem->buf = static_cast<u8 *>(sendRecvItem->buf) + curSize; // 更新usrIn，下一次从新的地址取数据
                if (sendRecvItem->count == 0){
                    emptyItem++;
                }
            }
        }
    }
    for (u32 i = 0; i < sendStreamNum_; i++) {
        const auto& sendQueueInner = sendDataSilcesBySendStream_[i];
        for (const auto& slice : sendQueueInner){
            HCCL_INFO("[CalcSendSlices] sendstream[%u] slice.addr[%p], slice.size[%llu] slice.remoteRank[%u]", i, slice.addr, slice.size, slice.remoteRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcRecvSlices()
{
    recvDataSilcesByRecvStream_.resize(recvStreamNum_);
    for (u32 i = 0; i < recvStreamNum_; i++) { // 按recvStream来放入dataSlice
        auto recvQueueInner = recvQueueByRecvstream_[i];
        // 遍历这个stream对应的queue中的所有item，每次取一个slice，然后再重新遍历，直到所有的itemcount减为0
        u32 emptyItem = 0;
        while (emptyItem < recvQueueInner.size()) {
            for (u32 j = 0; j < recvQueueInner.size(); j++){
                HcclSendRecvItem* sendRecvItem = recvQueueInner[j];
                if (sendRecvItem->count == 0) {
                    continue;
                }
                u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
                u64 maxCountPerLoop = CalcRecvLoopMaxCount(unitSize);
                u8 *curOutputPtr = static_cast<u8 *>(sendRecvItem->buf);
                CHK_PTR_NULL(curOutputPtr);
                u64 curCount = (sendRecvItem->count > maxCountPerLoop) ? maxCountPerLoop : sendRecvItem->count;
                u64 curSize = curCount * unitSize; // 单位：字节
                recvDataSilcesByRecvStream_[i].emplace_back(curOutputPtr, curSize, sendRecvItem->remoteRank);
                sendRecvItem->count -= curCount; // 更新剩余的count, 若sendRecvItem->count <= maxCountPerLoop，sendRecvItem->count会更新为0
                sendRecvItem->buf = static_cast<u8 *>(sendRecvItem->buf) + curSize; // 更新usrOut，下一次从新的地址取数据
                if (sendRecvItem->count == 0){
                    emptyItem++;
                }
            }
        }
    }
    for (u32 i = 0; i < recvStreamNum_; i++) {
        auto recvQueueInner = recvDataSilcesByRecvStream_[i];
        for (auto slice : recvQueueInner){
            HCCL_INFO("[CalcRecvSlices] recvstream[%u] slice.addr[%p], slice.size[%llu] slice.remoteRank[%u]", i, slice.addr, slice.size, slice.remoteRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcSendSlicesSmall()
{
    while (!sendDeque_.empty()) {
        HcclSendRecvItem* sendRecvItem = sendDeque_.front();
        HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcSendSlicesSmall] tag[%s], remoteRank[%u], buf[%p], count[%llu],"\
            "dataType[%s], sendRecvType[%d].", tag_.c_str(), sendRecvItem->remoteRank, sendRecvItem->buf,
            sendRecvItem->count, GetDataTypeEnumStr(sendRecvItem->dataType).c_str(), sendRecvItem->sendRecvType);
        u8 *curInputPtr = static_cast<u8 *>(sendRecvItem->buf);
        CHK_PTR_NULL(curInputPtr);
        u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
        u64 maxCountPerLoop = CalcSendLoopMaxCount(unitSize);

        for (u64 countLeft = sendRecvItem->count, curCount = 0, curOffset = 0; countLeft > 0;
            countLeft -= curCount) {
            curInputPtr += curOffset;
            curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
            u64 curSize = curCount * unitSize; // 单位：字节
            sendDataSilces_.emplace_back(curInputPtr, curSize, sendRecvItem->remoteRank);
            HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcSendSlicesSmall] slice userAddr[%p], slice size[%llu], remoteRank[%u]", curInputPtr, curSize, sendRecvItem->remoteRank);
            curOffset = curSize;
        }
        sendDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcRecvSlicesSmall()
{
    while (!recvDeque_.empty()) {
        HcclSendRecvItem* sendRecvItem = recvDeque_.front();
        HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcRecvSlicesSmall] tag[%s], remoteRank[%u], buf[%p], count[%llu],"\
            "dataType[%s], sendRecvType[%d].", tag_.c_str(), sendRecvItem ->remoteRank, sendRecvItem ->buf, sendRecvItem->count,
            GetDataTypeEnumStr(sendRecvItem->dataType).c_str(), sendRecvItem->sendRecvType);
        u8 *curOutputPtr = static_cast<u8*>(sendRecvItem->buf);
        CHK_PTR_NULL(curOutputPtr);
        u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
        u64 maxCountPerLoop = CalcRecvLoopMaxCount(unitSize);

        for (u64 countLeft = sendRecvItem->count, curCount = 0, curOffset = 0; countLeft > 0;
            countLeft -= curCount) {
            curOutputPtr += curOffset;
            curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
            u64 curSize = curCount * unitSize; // 单位：字节
            recvDataSilces_.emplace_back(curOutputPtr, curSize, sendRecvItem->remoteRank);
            HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcRecvSlicesSmall] slice userAddr[%p], slice size[%llu], remoteRank[%u]", curOutputPtr, curSize, sendRecvItem->remoteRank);
            curOffset = curSize;
        }
        recvDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::SubNotifyMain(Stream& stream, u32 streamId) const
{
    CHK_RET(LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesMain[streamId], PROF_STAGE_0));
    CHK_RET(LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesAux[streamId], PROF_STAGE_0));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::ProcessSendStreamDataSlice(Stream& stream, bool needStreamSync, 
    bool retryEnable, bool isSendStream, u32 sendStreamId)
{
    SendRecvSlice& slice = isSendStream ? sendDataSilcesBySendStream_[sendStreamId].front() : sendDataSilces_.front();
    u64 offset = bufferSliceSize_ * (slice.remoteRank % sendStreamNum_);
    DeviceMem inCommMem = algResResp_->cclInputMem.range(offset, slice.size);
    HCCL_INFO("[ProcessSendStreamDataSlice] inCommMem ptr[%p], size[%llu]", inCommMem.ptr(), inCommMem.size());
    if (!isSendStream) {
        DeviceMem inMem(slice.addr, slice.size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, stream));
    }
    if (needStreamSync) {
        if (isSendStream) {
            CHK_RET(SubNotifyMain(stream, sendStreamId));
        } else {
            CHK_RET(SubNotifyMain(stream, slice.remoteRank % sendStreamNum_));
        }
    }

    ExecMem execMem;
    execMem.inputMem = inCommMem;
    HcclResult ret = SendKernelRun(stream, execMem, slice.remoteRank, retryEnable);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBatchSendRecvGroupExecutor][ProcessSendStreamDataSlice]errNo[0x%016llx]kernel run error, tag[%s], " \
        "input_ptr[%p], size[%llu]", HCCL_ERROR_CODE(ret), tag_.c_str(), execMem.inputMem.ptr(),
        slice.size), ret);
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::ProcessDataSliceSmall(OpParam& param)
{
    CHK_RET(MainPostSubWaitSmall(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    while (!sendDataSilces_.empty() || !recvDataSilces_.empty()) {
        if(!sendDataSilces_.empty()) {
            CHK_RET(ProcessSendStreamDataSlice(param.stream, false, false));
            sendDataSilces_.pop_front();
        }
        if(!recvDataSilces_.empty()) {
            CHK_RET(ProcessRecvStreamDataSlice(algResResp_->slaveStreams[STREAM_INDEX_0], false, false));
            recvDataSilces_.pop_front();
        }
    }
    CHK_RET(MainWaitSubPostSmall(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::MainPostSubWaitSmall(Stream& mainStream, Stream& subStream) const
{
    CHK_RET(LocalNotify::Post(mainStream, dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_0], PROF_STAGE_0));
    CHK_RET(LocalNotify::Wait(subStream, dispatcher_,
        algResResp_->notifiesAux[STREAM_INDEX_0], PROF_STAGE_0));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::MainWaitSubPostSmall(Stream& mainStream, Stream& subStream) const
{
    CHK_RET(LocalNotify::Post(subStream, dispatcher_,
        algResResp_->notifiesMain[STREAM_INDEX_0], PROF_STAGE_0));

    CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_0],
        PROF_STAGE_0));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::ProcessRecvStreamDataSlice(Stream& stream, bool needStreamSync, bool retryEnable, bool isRecvStream, u32 recvStreamId)
{
    SendRecvSlice& slice = isRecvStream ? recvDataSilcesByRecvStream_[recvStreamId].front() :recvDataSilces_.front();
    
    ExecMem execMem;
    if (needStreamSync) {
        if (isRecvStream) {
            CHK_RET(SubNotifyMain(stream, recvStreamId + sendStreamNum_));
        } else {
            CHK_RET(SubNotifyMain(stream, slice.remoteRank % recvStreamNum_ + sendStreamNum_));
        }
    }
    LINK targetLink;
    CHK_RET(GetRecvTargetLink(slice.remoteRank, targetLink));
    if (topoAttr_.isDiffDeviceType || topoAttr_.superPodNum > 1 || 
            (topoAttr_.moduleNum > 1 && static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable())) ||
            (topoAttr_.deviceType == DevType::DEV_TYPE_910B && targetLink->GetLinkType() == LinkType::LINK_ROCE)) {
        u64 offset = bufferSliceSize_ * (slice.remoteRank % recvStreamNum_);
        execMem.outputMem = algResResp_->cclOutputMem.range(offset, slice.size);
        HcclResult ret = RecvKernelRun(stream, execMem, slice.remoteRank, retryEnable);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollBatchSendRecvGroupExecutor][ProcessRecvStreamDataSlice]errNo[0x%016llx]kernel run error, tag[%s], " \
            "output_ptr[%p], size[%llu]", HCCL_ERROR_CODE(ret), tag_.c_str(), execMem.outputMem.ptr(),
            slice.size), ret);

        DeviceMem outMem(slice.addr, slice.size);
        DeviceMem outCommMem = execMem.outputMem;
        HCCL_INFO("[ProcessRecvStreamDataSlice] outMem ptr[%p], size[%llu]", outMem.ptr(), outMem.size());
        HCCL_INFO("[ProcessRecvStreamDataSlice] outCommMem ptr[%p], size[%llu]", outCommMem.ptr(), outCommMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, stream));
    } else {
        DeviceMem outMem(slice.addr, slice.size);
        execMem.outputMem = outMem;
        HCCL_INFO("[ProcessRecvStreamDataSlice] DMA Reduce, outMem ptr[%p], size[%llu]", outMem.ptr(), outMem.size());
        HcclResult ret = RecvKernelRun(stream, execMem, slice.remoteRank, retryEnable);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollBatchSendRecvGroupExecutor][ProcessRecvStreamDataSlice]errNo[0x%016llx]kernel run error, tag[%s], " \
            "input_ptr[%p], size[%llu]", HCCL_ERROR_CODE(ret), tag_.c_str(), execMem.inputMem.ptr(),
            slice.size), ret);
    }
    return HCCL_SUCCESS;
}


u64 CollBatchSendRecvGroupExecutor::CalcSendLoopMaxCount(const u32 unitSize) const
{
    // 中转内存单次最多能够接受的input count
    u64 maxCountPerLoop = bufferSliceSize_ / unitSize;
    HCCL_WARNING("[CollBatchSendRecvGroupExecutor][CalcSendLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

u64 CollBatchSendRecvGroupExecutor::CalcRecvLoopMaxCount(const u32 unitSize) const
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = bufferSliceSize_ / unitSize;
    HCCL_WARNING("[CollBatchSendRecvGroupExecutor][CalcRecvLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcStreamNum(u32& streamNum)
{
    if (topoAttr_.userRankSize == 1) {
        sendStreamNum_ = 0;
        recvStreamNum_ = 0;
        streamNum = 0;
        HCCL_INFO("[CollBatchSendRecvGroupExecutor] Only one rank, do not need substream, streamNum[%u]", streamNum);
        return HCCL_SUCCESS;
    }
    sendStreamNum_ = GROUP_MAX_CONCURRENT;
    recvStreamNum_ = GROUP_MAX_CONCURRENT;
    streamNum = sendStreamNum_ + recvStreamNum_; // 有限度并发
    HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcStreamNum] tag_[%s], streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BatchSendRecvGroup", BatchSendRecvGroupExecutor, CollBatchSendRecvGroupExecutor);
} // namespace hccl