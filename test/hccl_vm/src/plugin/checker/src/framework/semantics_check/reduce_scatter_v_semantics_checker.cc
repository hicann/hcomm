/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Description: ReduceScatterV语义校验实现类
 */

#include "reduce_scatter_v_semantics_checker.h"

#include <map>

#include "base.h"
#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckReduceScatterVSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                            HcclReduceOp reduceType, VDataDesTagInner &vDataDes)
{
    u32 rankSize = allRankMemSemantics.size();
    u64 outputSize = 0;

    for (u32 rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR("{} ReduceScatterV produced no result data for rank {}, but this rank is "
                "expected to have one reduced shard, expectedSourceRankCount={}, expectedResultSize=0x{:x}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, rankSize,
                vDataDes.counts[rankId] * CHECK_SIZE_TABLE[vDataDes.dataType]);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR("{} ReduceScatterV result data does not start from the expected "
                    "address, rankId={}, expectedStartAddr=0x{:x}, actualStartAddr=0x{:x}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
                HCCL_VM_ERROR("{} Reduce mode of this ReduceScatterV result range does not match "
                    "the operator setting, rankId={}, actualReduceType={}, expectedReduceType={}, "
                    "outputRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), rankId,
                    DumpReduceOpToString(ele.reduceType), DumpReduceOpToString(reduceType),
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != rankSize) {
                HCCL_VM_ERROR("{} This ReduceScatterV result range does not include the expected "
                    "number of source ranks, rankId={}, actualSourceRankCount={}, expectedRankSize={}, "
                    "outputRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), rankId, ele.srcBufs.size(),
                    rankSize, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            if (ele.srcBufs.begin()->rankId != 0 or ele.srcBufs.rbegin()->rankId != static_cast<int>(rankSize - 1)) {
                HCCL_VM_ERROR("{} Source rank list for this ReduceScatterV result range is not the "
                    "complete expected rank set, rankId={}, firstSourceRank={}, lastSourceRank={}, "
                    "expectedFirstSourceRank=0, expectedLastSourceRank={}, outputRange=[0x{:x},0x{:x})"
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId,
                    ele.srcBufs.begin()->rankId, ele.srcBufs.rbegin()->rankId, rankSize - 1, ele.startAddr,
                    rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            for (auto &srcBuf : ele.srcBufs) {
                if (srcBuf.bufType != BufferType::INPUT) {
                    HCCL_VM_ERROR("{} This ReduceScatterV result range comes from a non-INPUT "
                        "buffer, but it should come from INPUT, rankId={}, actualSourceRank={}, "
                        "actualSourceBufferType={}, actualBufferRange=[0x{:x},0x{:x})"
                        "\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId, srcBuf.rankId,
                        BufferTypeToString(srcBuf.bufType), ele.startAddr, rangeEnd, ele.Describe());
                    HcclResult::HCCL_E_PARA;
                }

                const u64 outputOffset = vDataDes.displs[rankId] * CHECK_SIZE_TABLE[vDataDes.dataType];
                if (srcBuf.srcAddr != outputOffset + totalSize) {
                    HCCL_VM_ERROR("{} Source address for this ReduceScatterV result range does not "
                        "match the expected input address, rankId={}, sourceRank={}, expectedAddr=0x{:x}, "
                        "actualAddr=0x{:x}, actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), rankId, srcBuf.rankId,
                        outputOffset + totalSize, srcBuf.srcAddr, ele.startAddr, rangeEnd, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
            }
            totalSize += ele.size;
        }
        outputSize = vDataDes.counts[rankId] * CHECK_SIZE_TABLE[vDataDes.dataType];

        if (totalSize != outputSize) {
            HCCL_VM_ERROR("{} ReduceScatterV result data ends before the expected total size is "
                "reached, rankId={}, checkedSize=0x{:x}, expectedSize=0x{:x}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
