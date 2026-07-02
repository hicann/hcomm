/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_all_rank_param_recorder.h"

#include <cstdint>
#include <sys/types.h>

namespace HcclSim {
AllRankParamRecorder* AllRankParamRecorder::Global()
{
    static AllRankParamRecorder* allRankParamRecorder = new AllRankParamRecorder;
    return allRankParamRecorder;
}

void AllRankParamRecorder::Reset()
{
    curXn.clear();
    curGSA.clear();
    curCKE.clear();
    curHBM.clear();
    seenPost.clear();
    return;
}

void AllRankParamRecorder::InitParam()
{
    return;
}

HcclResult AllRankParamRecorder::SetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t xnValue)
{
    curXn[rankId][dieId][xnId] = xnValue;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::SetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t gsaValue)
{
    curGSA[rankId][dieId][gsaId] = gsaValue;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::SetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t ckeValue)
{
    curCKE[rankId][dieId][ckeId] = ckeValue;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::GetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t& xnValue)
{
    if(curXn.find(rankId) != curXn.end()) {
        if (curXn[rankId].find(dieId) != curXn[rankId].end()) {
            if (curXn[rankId][dieId].find(xnId) != curXn[rankId][dieId].end()) {
                xnValue = curXn[rankId][dieId][xnId];
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[GetXn]curXn is not be initialized, rankId[%u], dieId[%u], xnId[%hu]", rankId, dieId, xnId);
    return HCCL_E_PARA;
}

HcclResult  AllRankParamRecorder::GetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t& gsaValue)
{
    if(curGSA.find(rankId) != curGSA.end()) {
        if (curGSA[rankId].find(dieId) != curGSA[rankId].end()) {
            if (curGSA[rankId][dieId].find(gsaId) != curGSA[rankId][dieId].end()) {
                gsaValue = curGSA[rankId][dieId][gsaId];
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[GetGSA]curGSA is not be initialized, rankId[%u], dieId[%u], gsaId[%hu]", rankId, dieId, gsaId);
    return HCCL_E_PARA;
}

HcclResult AllRankParamRecorder::GetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t& ckeValue)
{
    if(curCKE.find(rankId) != curCKE.end()) {
        if (curCKE[rankId].find(dieId) != curCKE[rankId].end()) {
            if (curCKE[rankId][dieId].find(ckeId) != curCKE[rankId][dieId].end()) {
                ckeValue = curCKE[rankId][dieId][ckeId];
                return HCCL_SUCCESS;
            }
        }
    }

    ckeValue = 0;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::CheckAllPostMatch()
{
    for (const auto& rankPair : seenPost) {
        RankId rank = rankPair.first;
        for (const auto& diePair : rankPair.second) {
            uint32_t dieId = diePair.first;
            for (const auto& regPair: diePair.second) {
                uint16_t regId = regPair.first;
                for (auto& post : seenPost[rank][dieId][regId]) {
                    HCCL_WARNING("unmatched LocalPost/Post: %s", post->task->Describe().c_str());
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::SetHBM(uint32_t rankId, uint32_t dieId, uint64_t hbmAddr,
    const std::vector<uint64_t>& data) {
    // 合法性校验
    if (data.size() % 8 != 0){
        HCCL_ERROR("hbm data size is not 8 bytes aligned");
        return HCCL_E_PARA;
    }
    curHBM[rankId][dieId][hbmAddr] = data;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::GetHBM(uint32_t rankId, uint32_t dieId, uint64_t hbmAddr, std::vector<uint64_t>& data) {
    if(curHBM.find(rankId) != curHBM.end()) {
        if (curHBM[rankId].find(dieId) != curHBM[rankId].end()) {
            if (curHBM[rankId][dieId].find(hbmAddr) != curHBM[rankId][dieId].end()) {
                data = curHBM[rankId][dieId][hbmAddr];
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[GetGSA]curGSA is not be initialized, rankId[%u], dieId[%u], hbmAddr[%llu]", rankId, dieId, hbmAddr);
    return HCCL_E_PARA;
}

}
