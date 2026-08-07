/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu context header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_CONTEXT_ASSIST_H
#define HCOMM_CCU_CONTEXT_ASSIST_H

#include <string>

#include "ccu_assist_pub.h"

#include "data_type.h"
#include "reduce_op.h"

namespace hcomm {
namespace CcuRep {

    // 辅助函数
    uint64_t GetToken(uint64_t tokenId, uint64_t tokenValue, uint64_t tokenValid);

    // 按 token 寄存器位域规则将 (tokenId, tokenValue, tokenValid) 三元组合成单个 64bit token 字段。
    // 与 GetToken 行为完全一致，提供更贴近 C API 调用方语义的命名，避免在 C 适配层重复实现位域逻辑。
    uint64_t CcuCombineTokenInfo(uint64_t tokenId, uint64_t tokenValue, uint64_t tokenValid);

    uint16_t GetCcuReduceType(Hccl::ReduceOp reduceOp);
    uint16_t GetCcuDataType(Hccl::DataType dataType, Hccl::ReduceOp reduceOp);
    uint16_t GetUBReduceType(Hccl::ReduceOp reduceOp);
    uint16_t GetUBDataType(Hccl::DataType dataType);

    uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum);
    uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum);
    uint64_t GetParallelParamV2(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum);
    uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset);

}; // namespace CcuRep
}; // namespace hcomm

#endif // HCCL_CCU_CONTEXT_ASSIST_H
