/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_recv_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckSendRecvSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize,
                                      RankId srcRank, RankId dstRank)
{
    u32 rankSize = allRankMemSemantics.size();

    // 对应的rank不存在需要报错
    if (allRankMemSemantics.size() != 2) {
        HCCL_VM_ERROR("{} Send/Recv final output validation supports exactly 2 ranks, but got "
            "expectedRankSize=2, actualRankSize={}, sourceRank={}, targetRank={}",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_CHECK_FAILED), rankSize, srcRank, dstRank);
        return HcclResult::HCCL_E_PARA;
    }

    u64 totalSize = 0;
    for (auto &ele : allRankMemSemantics[dstRank][BufferType::OUTPUT]) {
        const u64 rangeEnd = ele.startAddr + ele.size;
        if (ele.startAddr != totalSize) {
            HCCL_VM_ERROR("{} Send/Recv result data does not start from the expected address, "
                "rankId={}, expectedStartAddr=0x{:x}, actualStartAddr=0x{:x}, actualBufferRange=[0x{:x},0x{:x})"
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), dstRank, totalSize, ele.startAddr,
                ele.startAddr, rangeEnd, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() != 1) {
            HCCL_VM_ERROR("{} This Send/Recv result range combines multiple sources, but this "
                "operator expects exactly one source, rankId={}, actualSourceCount={}, expectedSourceCount=1, "
                "outputRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), dstRank, ele.srcBufs.size(),
                ele.startAddr, rangeEnd, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.begin()->rankId != srcRank) {
            HCCL_VM_ERROR("{} This Send/Recv result range comes from the wrong source rank, "
                "rankId={}, actualSourceRank={}, expectedSourceRank={}, actualBufferRange=[0x{:x},0x{:x})"
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), dstRank,
                ele.srcBufs.begin()->rankId, srcRank, ele.startAddr, rangeEnd, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
            HCCL_VM_ERROR("{} This Send/Recv result range comes from a non-INPUT buffer, but it "
                "should come from INPUT, rankId={}, actualSourceRank={}, actualSourceBufferType={}, "
                "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), dstRank,
                ele.srcBufs.begin()->rankId, BufferTypeToString(ele.srcBufs.begin()->bufType),
                ele.startAddr, rangeEnd, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }
        if (ele.srcBufs.begin()->srcAddr != totalSize) {
            HCCL_VM_ERROR("{} Source address for this Send/Recv result range does not match the "
                "expected input address, rankId={}, sourceRank={}, expectedAddr=0x{:x}, actualAddr=0x{:x}, "
                "actualBufferRange=[0x{:x},0x{:x})\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), dstRank, ele.srcBufs.begin()->rankId,
                totalSize, ele.srcBufs.begin()->srcAddr, ele.startAddr, rangeEnd, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }
        totalSize += ele.size;
    }
    if (totalSize != dataSize) {
        HCCL_VM_ERROR("{} Send/Recv result data ends before the expected total size is reached, "
            "rankId={}, checkedSize=0x{:x}, expectedSize=0x{:x}",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), dstRank, totalSize, dataSize);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
