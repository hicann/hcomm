/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BATCH_SEND_RECV_GROUP_EXECUTOR_H
#define COLL_BATCH_SEND_RECV_GROUP_EXECUTOR_H

#include "coll_comm_executor.h"
#include "coll_batch_send_recv_executor.h"

namespace hccl {
class CollBatchSendRecvGroupExecutor : public CollBatchSendRecvExecutor {
public:
    CollBatchSendRecvGroupExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollBatchSendRecvGroupExecutor() override = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algResource) override;
protected:

    /* *************** 算法编排 *************** */
    u64 CalcSendLoopMaxCount(const u32 unitSize) const;
    u64 CalcRecvLoopMaxCount(const u32 unitSize) const;
    HcclResult ProcessSendStreamDataSlice(Stream& stream, bool needStreamSync, bool retryEnable, bool isSendStream = false, u32 sendStreamId = 0);
    HcclResult ProcessRecvStreamDataSlice(Stream& stream, bool needStreamSync, bool retryEnable, bool isRecvStream = false, u32 recvStreamId = 0);
    HcclResult ProcessDataSliceSmall(OpParam& param);
    HcclResult CalcSendSlices();
    HcclResult CalcRecvSlices();
    HcclResult CalcSendSlicesSmall();
    HcclResult CalcRecvSlicesSmall();
    HcclResult CopyLocalDataForARound(Stream& mainStream);
    HcclResult OrganizeSendSlices();
    HcclResult OrganizeRecvSlices();
    HcclResult OrganizeSendItemByStream();
    HcclResult OrganizeRecvItemByStream();
    struct SendRecvSlice {
        u8* addr;
        u64 size;
        u32 remoteRank;
        SendRecvSlice(u8* addr, u64 size, u32 remoteRank) : addr(addr), size(size), remoteRank(remoteRank) {}
    };

    std::deque<SendRecvSlice> sendDataSilces_;
    std::deque<SendRecvSlice> recvDataSilces_;
    std::vector<std::deque<SendRecvSlice>> sendDataSilcesBySendStream_;
    std::vector<std::deque<SendRecvSlice>> recvDataSilcesByRecvStream_;

private:
    HcclResult RunLoopBig(OpParam& param);
    HcclResult RunTasksBig(OpParam& param);
    HcclResult RunLoopSmall(OpParam& param);
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult isGroupBigCount(HcclSendRecvItem *sendRecvInfo, u32 itemNum, bool& isBig) const;
    HcclResult CalcBufferSliceSize();

    HcclResult MainPostSubWait(Stream& mainStream);
    HcclResult MainWaitSubPost(Stream& mainStream);
    HcclResult SubNotifyMain(Stream& stream, u32 streamId) const;
    HcclResult MainPostSubWaitSmall(Stream& mainStream, Stream& subStream) const;
    HcclResult MainWaitSubPostSmall(Stream& mainStream, Stream& subStream) const;

private:

    std::vector<std::deque<HcclSendRecvItem*>> sendQueueBySendstream_;
    std::vector<std::deque<HcclSendRecvItem*>> recvQueueByRecvstream_;
    u32 sendStreamNum_ = 0;
    u32 recvStreamNum_ = 0;
    u64 bufferSliceSize_ = 0;
};
} // namespace hccl

#endif