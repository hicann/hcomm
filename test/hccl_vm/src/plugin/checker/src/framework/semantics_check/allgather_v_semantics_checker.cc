/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Description: AllGatherV语义校验实现类
 */

#include "allgather_v_semantics_checker.h"

#include <map>

#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckAllGatherVSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, VDataDesTagInner &vDataDes)
{
    u32 rankSize = allRankMemSemantics.size();

    u64 outputSize = 0;
    // AllGatherV 输入不等长
    for (u32 i = 0; i < rankSize; i++) {
        u64 curCounts = vDataDes.counts[i];
        u64 curLength = curCounts * CHECK_SIZE_TABLE[vDataDes.dataType];
        outputSize += curLength;
    }

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR("{} AllGatherV produced no result data for rank {}, but this rank is "
                "expected to receive data from all ranks, expectedSourceRankCount={}, expectedResultSize=0x{:x}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, rankSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }

        u64    totalSize   = 0;
        RankId curRankId   = 0;
        u64    curDataSize = 0;
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            u64 inputSize = vDataDes.counts[curRankId] * CHECK_SIZE_TABLE[vDataDes.dataType];
            while (!inputSize) {
                curRankId++;
                inputSize = vDataDes.counts[curRankId] * CHECK_SIZE_TABLE[vDataDes.dataType];
            }
            const u64 rangeEnd = ele.startAddr + ele.size;

            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR("{} AllGatherV result data does not start from the expected address, "
                    "rankId={}, expectedStartAddr=0x{:x}, actualStartAddr=0x{:x}, actualBufferRange=[0x{:x},0x{:x})"
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR("{} This AllGatherV result range combines multiple sources, but this "
                    "operator expects exactly one source, rankId={}, actualSourceCount={}, expectedSourceCount=1, "
                    "outputRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), rankId, ele.srcBufs.size(),
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->rankId != curRankId) {
                HCCL_VM_ERROR("{} This AllGatherV result range comes from the wrong source rank, "
                    "rankId={}, actualSourceRank={}, expectedSourceRank={}, actualBufferRange=[0x{:x},0x{:x})"
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, curRankId, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
                HCCL_VM_ERROR("{} This AllGatherV result range comes from a non-INPUT buffer, but "
                    "it should come from INPUT, rankId={}, actualSourceRank={}, actualSourceBufferType={}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, BufferTypeToString(ele.srcBufs.begin()->bufType),
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->srcAddr != curDataSize) {
                HCCL_VM_ERROR("{} Source address for this AllGatherV result range does not match "
                    "the expected input address, rankId={}, sourceRank={}, expectedAddr=0x{:x}, actualAddr=0x{:x}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, curDataSize, ele.srcBufs.begin()->srcAddr, ele.startAddr,
                    rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;

            if (curDataSize == inputSize) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > inputSize) {
                HCCL_VM_ERROR("{} Data taken from one source rank is larger than expected in "
                    "AllGatherV, rankId={}, sourceRank={}, actualSize=0x{:x}, expectedSize=0x{:x}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), rankId, curRankId, curDataSize,
                    inputSize, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != outputSize) {
            HCCL_VM_ERROR("{} AllGatherV result data ends before the expected total size is "
                "reached, rankId={}, checkedSize=0x{:x}, expectedSize=0x{:x}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
