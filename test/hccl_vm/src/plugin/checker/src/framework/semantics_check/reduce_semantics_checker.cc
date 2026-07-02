/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckReduceSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize,
                                    HcclReduceOp reduceType, RankId root)
{
    u32 rankSize = allRankMemSemantics.size();
    if (rankSize == 0) {
        return HCCL_SUCCESS;
    }

    // 对应的rank不存在需要报错
    if (allRankMemSemantics.count(root) == 0 && dataSize > 0) {
        HCCL_VM_ERROR("{} Reduce produced no result data for root rank {}, but this rank is "
            "expected to hold the reduced result, expectedSourceRankCount={}, expectedResultSize=0x{:x}",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), root, rankSize, dataSize);
        return HcclResult::HCCL_E_PARA;
    }

    u64 totalSize = 0;
    for (auto &ele : allRankMemSemantics[root][BufferType::OUTPUT]) {
        const u64 rangeEnd = ele.startAddr + ele.size;
        if (ele.startAddr != totalSize) {
            HCCL_VM_ERROR("{} Reduce result data does not start from the expected address, "
                "rankId={}, expectedStartAddr=0x{:x}, actualStartAddr=0x{:x}, actualBufferRange=[0x{:x},0x{:x})"
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), root, totalSize, ele.startAddr,
                ele.startAddr, rangeEnd, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
            HCCL_VM_ERROR("{} Reduce mode of this Reduce result range does not match the operator "
                "setting, rankId={}, actualReduceType={}, expectedReduceType={}, outputRange=[0x{:x},0x{:x})"
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), root,
                DumpReduceOpToString(ele.reduceType), DumpReduceOpToString(reduceType), ele.startAddr, rangeEnd,
                ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() != rankSize) {
            HCCL_VM_ERROR("{} This Reduce result range does not include the expected number of "
                "source ranks, rankId={}, actualSourceRankCount={}, expectedRankSize={}, "
                "outputRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), root, ele.srcBufs.size(), rankSize,
                ele.startAddr, rangeEnd, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.begin()->rankId != 0 or ele.srcBufs.rbegin()->rankId != (rankSize - 1)) {
            HCCL_VM_ERROR("{} Source rank list for this Reduce result range is not the complete "
                "expected rank set, rankId={}, firstSourceRank={}, lastSourceRank={}, expectedFirstSourceRank=0, "
                "expectedLastSourceRank={}, outputRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), root,
                ele.srcBufs.begin()->rankId, ele.srcBufs.rbegin()->rankId, rankSize - 1, ele.startAddr, rangeEnd,
                ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        for (auto &srcBuf : ele.srcBufs) {
            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR("{} This Reduce result range comes from a non-INPUT buffer, but it "
                    "should come from INPUT, rankId={}, actualSourceRank={}, actualSourceBufferType={}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), root, srcBuf.rankId,
                    BufferTypeToString(srcBuf.bufType), ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != totalSize) {
                HCCL_VM_ERROR("{} Source address for this Reduce result range does not match the "
                    "expected input address, rankId={}, sourceRank={}, expectedAddr=0x{:x}, actualAddr=0x{:x}, "
                    "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), root, srcBuf.rankId, totalSize,
                    srcBuf.srcAddr, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
        }
        totalSize += ele.size;
    }
    if (totalSize != dataSize) {
        HCCL_VM_ERROR("{} Reduce result data ends before the expected total size is reached, "
            "rankId={}, checkedSize=0x{:x}, expectedSize=0x{:x}",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), root, totalSize, dataSize);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
