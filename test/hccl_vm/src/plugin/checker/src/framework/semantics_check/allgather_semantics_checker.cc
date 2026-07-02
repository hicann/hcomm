/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_semantics_checker.h"

#include <map>

#include "check_utils.h"
#include "sim_common.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckAllGatherSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize)
{
    u32 rankSize = allRankMemSemantics.size();

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR("{} AllGather produced no result data for rank {}, but this rank is "
                "expected to receive data from all ranks, expectedSourceRankCount={}, expectedResultSize=0x{:x}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, rankSize, dataSize * rankSize);
            return HcclResult::HCCL_E_PARA;
        }

        u64    totalSize   = 0;
        RankId curRankId   = 0;
        u64    curDataSize = 0;
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR("{} AllGather result data does not start from the expected address, "
                    "rankId={}, expectedStartAddr=0x{:x}, actualStartAddr=0x{:x}, actualBufferRange=[0x{:x},0x{:x})"
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR("{} This AllGather result range combines multiple sources, but this "
                    "operator expects exactly one source, rankId={}, actualSourceCount={}, expectedSourceCount=1, "
                    "outputRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), rankId, ele.srcBufs.size(),
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->rankId != curRankId) {
                HCCL_VM_ERROR("{} This AllGather result range comes from the wrong source rank, "
                    "rankId={}, actualSourceRank={}, expectedSourceRank={}, actualBufferRange=[0x{:x},0x{:x})"
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, curRankId, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
                HCCL_VM_ERROR("{} This AllGather result range comes from a non-INPUT buffer, but it "
                    "should come from INPUT, rankId={}, actualSourceRank={}, actualSourceBufferType={}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, BufferTypeToString(ele.srcBufs.begin()->bufType),
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->srcAddr != curDataSize) {
                HCCL_VM_ERROR("{} Source address for this AllGather result range does not match the "
                    "expected input address, rankId={}, sourceRank={}, expectedAddr=0x{:x}, actualAddr=0x{:x}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, curDataSize, ele.srcBufs.begin()->srcAddr, ele.startAddr,
                    rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            if (curDataSize == dataSize) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > dataSize) {
                HCCL_VM_ERROR("{} Data taken from one source rank is larger than expected in "
                    "AllGather, rankId={}, sourceRank={}, actualSize=0x{:x}, expectedSize=0x{:x}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), rankId, curRankId, curDataSize,
                    dataSize, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize * rankSize) {
            HCCL_VM_ERROR("{} AllGather result data ends before the expected total size is "
                "reached, rankId={}, checkedSize=0x{:x}, expectedSize=0x{:x}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, dataSize * rankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
