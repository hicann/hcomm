/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rdma_conn_lite_v2.h"
#include "rdma_vendor_1825_ops.h"

namespace Hccl {
constexpr u64 RDMA_DMA_MAX_SIZE = 0x80000000; // Byte, RDMA一次传输的最大size

void RdmaConnLiteV2::ParseSqContext(std::vector<char>& data)
{
    BinaryStream binaryStream(data);
    binaryStream >> sqContext.qpn;
    binaryStream >> sqContext.sqVa;
    binaryStream >> sqContext.wqeSize;
    binaryStream >> sqContext.depth;
    binaryStream >> sqContext.headAddr;
    binaryStream >> sqContext.tailAddr;
    binaryStream >> sqContext.dbVa;
    binaryStream >> sqContext.dbMode;
    binaryStream >> sqContext.sl;
}

void RdmaConnLiteV2::ParseCqContext(std::vector<char>& data)
{
    BinaryStream binaryStream(data);
    binaryStream >> cqContext.cqn;
    binaryStream >> cqContext.cqVa;
    binaryStream >> cqContext.cqeSize;
    binaryStream >> cqContext.cqDepth;
    binaryStream >> cqContext.headAddr;
    binaryStream >> cqContext.tailAddr;
    binaryStream >> cqContext.dbVa;
    binaryStream >> cqContext.dbMode;
}

RdmaConnLiteV2::RdmaConnLiteV2(std::vector<char>& uniqueId) : RmaConnLite()
{
    BinaryStream binaryStream(uniqueId);
    binaryStream >> dmaMode_;

    std::vector<char> sqUniqueId;
    binaryStream >> sqUniqueId;
    ParseSqContext(sqUniqueId);

    std::vector<char> cqUniqueId;
    binaryStream >> cqUniqueId;
    ParseCqContext(cqUniqueId);

    qpVa_ = sqContext.sqVa;
    sqVa_ = sqContext.sqVa;
    sqDepth_ = sqContext.depth;

    // 确定厂商
    GetVendorOps();
}

RdmaConnLiteV2::~RdmaConnLiteV2() {}

std::string RdmaConnLiteV2::Describe()
{
    return StringFormat(
        "RdmaConnLiteV2[QPN=%u, SQ_VA=0x%llx, WQE_SIZE=%u, SQ_DEPTH=%u, SQ_HEAD_ADDR=0x%llx, SQ_TAIL_ADDR=0x%llx, "
        "SL=%u, SQ_DB_VA=0x%llx, SQ_DB_MODE=%d, CQN=%u, CQ_VA=0x%llx, CQE_SIZE=%u, CQ_DEPTH=%u, "
        "CQ_HEAD_ADDR=0x%llx, CQ_TAIL_ADDR=0x%llx, CQ_DB_VA=0x%llx, CQ_DB_MODE=%d]",
        sqContext.qpn, sqContext.sqVa, sqContext.wqeSize, sqContext.depth, sqContext.headAddr, sqContext.tailAddr,
        sqContext.sl, sqContext.dbVa, sqContext.dbMode,
        cqContext.cqn, cqContext.cqVa, cqContext.cqeSize, cqContext.cqDepth,
        cqContext.headAddr, cqContext.tailAddr, cqContext.dbVa, cqContext.dbMode
    );
}

void RdmaConnLiteV2::GetVendorOps()
{
    if (rdmaOps_ != nullptr) {
        return;
    }
    switch (dmaMode_) {
        case 0 : {   // [PCIe] QBUF_DMA_MODE_DEFAULT
            HCCL_INFO("[RdmaConnLiteV2::%s] Now Aicpu NDA doesn't support PCIE !", __func__);
            rdmaOps_ = nullptr;
            break;
        }
        case 1: {  // [UB] QBUF_DMA_MODE_INDEP_UB
            rdmaOps_ = std::make_unique<Rdma1825Ops>(&sqContext, &cqContext);
            break;
        }
        default: {
            HCCL_INFO("[RdmaConnLiteV2::%s] Now dmaMode is invalid !", __func__);
            rdmaOps_ = nullptr;
            break;
        }
    }
}

// 检查Ops不能为空
void RdmaConnLiteV2::CheckVendorOp()
{
    if (UNLIKELY(rdmaOps_ == nullptr)) {
        THROW<InternalException>(StringFormat("NDA Op is null. Now dmaMode_ is %d.", dmaMode_));
    }
}

void RdmaConnLiteV2::Write(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, u64 &dbAddr, u64 &dbValue)
{
    HCCL_INFO("[RdmaConnLiteV2::%s] Write start, loc size = %u", __func__, loc.GetSize());
    CheckVendorOp();
    
    // 分片write
    DoSliceWrite(loc, rmt);

    // 构造Doorbell并返回
    rdmaOps_->BuildDoorbell(dbAddr, dbValue);

    HCCL_INFO("[RdmaConnLiteV2::%s] end, dbAddr = %llu, dbValue = %llu, conn[%s]", __func__, dbAddr, dbValue, Describe().c_str());
}

void RdmaConnLiteV2::WriteWithNotify(
    const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt,
    const RmaBufSliceLite &locNotify, const RmtRmaBufSliceLite &notify, u64 &dbAddr, u64 &dbValue)
{
    HCCL_INFO("[RdmaConnLiteV2::%s] WriteWithNotify start, loc size = %u", __func__, loc.GetSize());
    CheckVendorOp();

    // 分片write
    DoSliceWrite(loc, rmt);

    // 补充一个notify操作
    rdmaOps_->NotifyRecord(locNotify, notify);

    // 构造Doorbell并返回
    rdmaOps_->BuildDoorbell(dbAddr, dbValue);

    HCCL_INFO("[RdmaConnLiteV2::%s] end, dbAddr = %llu, dbValue = %llu, conn[%s]", __func__, dbAddr, dbValue, Describe().c_str());
}

void RdmaConnLiteV2::DoSliceWrite(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt)
{
    const u64 len = loc.GetSize();
    const u32 fullSlices = static_cast<u32>(len / RDMA_DMA_MAX_SIZE);
    const u32 remain     = static_cast<u32>(len % RDMA_DMA_MAX_SIZE);
    const u32 totalSlices = fullSlices + (remain > 0 ? 1 : 0);

    for (u32 sliceIdx = 0; sliceIdx < totalSlices; sliceIdx++) {
        const u64 offset     = static_cast<u64>(sliceIdx) * RDMA_DMA_MAX_SIZE;
        const u64 localAddr  = loc.GetAddr() + offset;
        const u64 remoteAddr = rmt.GetAddr() + offset;
        const u32 sliceSize  = (sliceIdx == totalSlices - 1 && remain > 0)
                               ? remain : RDMA_DMA_MAX_SIZE;

        RmaBufSliceLite    locSlice(localAddr, sliceSize, loc.GetLkey(), 0);
        RmtRmaBufSliceLite rmtSlice(remoteAddr, sliceSize, rmt.GetRkey(), 0, 0, UINT32_MAX);

        HCCL_INFO("[RdmaConnLiteV2::%s] Slice[%u]: offset=0x%llx, localAddr=0x%llx, "
                  "remoteAddr=0x%llx, size=0x%x",
                  __func__, sliceIdx, offset, localAddr, remoteAddr, sliceSize);

        rdmaOps_->Write(locSlice, rmtSlice);
    }
}

} // namespace Hccl
