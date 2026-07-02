/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Description: All2All语义校验函数实现
 * Author: yinding
 * Create: 2024-09-30
 */

#include "all2all_semantics_checker.h"

#include <map>
#include <vector>

#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
void GenSendOffsetForOneRank(All2AllDataDesTagInner &all2AllDataDes, u32 rankSize, RankId targetRank, std::vector<u64> &sendOffsets)
{
    for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
        u64 offset = 0;
        for (RankId dstRank = 0; dstRank < targetRank; dstRank++) {
            u64 count = all2AllDataDes.sendCountMatrix[srcRank * rankSize + dstRank];
            u64 size = count * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            offset += size;
        }
        sendOffsets.push_back(offset);
    }
    return;
}

HcclResult TaskCheckAll2AllSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                     All2AllDataDesTagInner &all2AllDataDes)
{
    u32 rankSize = allRankMemSemantics.size();

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR("{} AllToAll produced no result data for rank {}, but this rank is "
                "expected to receive data from peer ranks, expectedSourceRankCount={}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, rankSize);
            return HcclResult::HCCL_E_PARA;
        }

        std::vector<u64> sendOffsets;
        GenSendOffsetForOneRank(all2AllDataDes, rankSize, rankId, sendOffsets);

        u64    totalSize   = 0;
        RankId curRankId   = 0;
        u64    curDataSize = 0;
 
        // curRankId向rankId发送的数据量是0 直接跳过
        while (curRankId < rankSize && all2AllDataDes.sendCountMatrix[curRankId * rankSize + rankId] == 0) {
            curRankId++;
        }
 
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR("{} AllToAll result data does not start from the expected address, "
                    "rankId={}, expectedStartAddr=0x{:x}, actualStartAddr=0x{:x}, actualBufferRange=[0x{:x},0x{:x})"
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR("{} This AllToAll result range combines multiple sources, but this "
                    "operator expects exactly one source, rankId={}, actualSourceCount={}, expectedSourceCount=1, "
                    "outputRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), rankId, ele.srcBufs.size(),
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (curRankId >= rankSize) {
                HCCL_VM_ERROR("{} Extra output data exists after the expected AllToAll result "
                    "range, rankId={}, extraBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), rankId, ele.startAddr, rangeEnd,
                    ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
 
            if (ele.srcBufs.begin()->rankId != curRankId) {
                HCCL_VM_ERROR("{} This AllToAll result range comes from the wrong source rank, "
                    "rankId={}, actualSourceRank={}, expectedSourceRank={}, actualBufferRange=[0x{:x},0x{:x})"
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, curRankId, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
                HCCL_VM_ERROR("{} This AllToAll result range comes from a non-INPUT buffer, but it "
                    "should come from INPUT, rankId={}, actualSourceRank={}, actualSourceBufferType={}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, BufferTypeToString(ele.srcBufs.begin()->bufType),
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->srcAddr != sendOffsets[curRankId] + curDataSize) {
                HCCL_VM_ERROR("{} Source address for this AllToAll result range does not match the "
                    "expected input address, rankId={}, sourceRank={}, expectedAddr=0x{:x}, actualAddr=0x{:x}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, sendOffsets[curRankId] + curDataSize,
                    ele.srcBufs.begin()->srcAddr, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            u64 recvCountFromCurRank = all2AllDataDes.sendCountMatrix[curRankId * rankSize + rankId];
            u64 recvDataSizeFromCurRank = recvCountFromCurRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            if (curDataSize == recvDataSizeFromCurRank) {
                curDataSize = 0;
                curRankId++;
                while (curRankId < rankSize && all2AllDataDes.sendCountMatrix[curRankId * rankSize + rankId] == 0) {
                    // 跳过后续所有发送量为0的rank
                    curRankId++;
                }
            } else if (curDataSize > recvDataSizeFromCurRank) {
                HCCL_VM_ERROR("{} Data taken from one source rank is larger than expected in "
                    "AllToAll, rankId={}, sourceRank={}, actualSize=0x{:x}, expectedSize=0x{:x}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), rankId, curRankId, curDataSize,
                    recvDataSizeFromCurRank, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankId等于rankSize，表示已经接受到其他所有rank的数据
        if (curRankId != rankSize) {
            HCCL_VM_ERROR("{} AllToAll result data ends before the expected total size is reached, "
                "rankId={}, checkedSize=0x{:x}, remainingSourceRank={}, remainingSizeFromThatRank=0x{:x}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, curRankId, curDataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
