/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_all_rank_param_recorder_v3.h"
#include <sys/types.h>
#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

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
    postNodeMeta.clear();
    devType_ = DevType::DEV_TYPE_COUNT;
    ccu_resource_base_addr_.clear();
    return;
}

void AllRankParamRecorder::InitParam()
{
    return;
}

void AllRankParamRecorder::RegisterPostNode(TaskNode *node, const CcuPostNodeMetaV3 &meta)
{
    if (node == nullptr) {
        return;
    }
    postNodeMeta[node] = meta;
}

uint32_t AllRankParamRecorder::GetPostRemainingCkeMask(const TaskNode *node) const
{
    auto it = postNodeMeta.find(node);
    if (it == postNodeMeta.end()) {
        return 0;
    }
    return it->second.remainingCkeMask;
}

void AllRankParamRecorder::SetPostRemainingCkeMask(TaskNode *node, uint32_t remainingCkeMask)
{
    auto it = postNodeMeta.find(node);
    if (it == postNodeMeta.end()) {
        return;
    }
    it->second.remainingCkeMask = static_cast<uint16_t>(remainingCkeMask);
}

const CcuPostNodeMetaV3 *AllRankParamRecorder::GetPostNodeMeta(const TaskNode *node) const
{
    auto it = postNodeMeta.find(node);
    if (it == postNodeMeta.end()) {
        return nullptr;
    }
    return &it->second;
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
    xnValue = 0;
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
    gsaValue = 0;
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

std::map<uint16_t, uint64_t> AllRankParamRecorder::GetXnSnapshot(uint32_t rankId, uint32_t dieId) const
{
    auto rankIt = curXn.find(rankId);
    if (rankIt == curXn.end()) {
        return {};
    }
    auto dieIt = rankIt->second.find(dieId);
    if (dieIt == rankIt->second.end()) {
        return {};
    }
    return dieIt->second;
}

std::map<uint16_t, uint64_t> AllRankParamRecorder::GetGSASnapshot(uint32_t rankId, uint32_t dieId) const
{
    auto rankIt = curGSA.find(rankId);
    if (rankIt == curGSA.end()) {
        return {};
    }
    auto dieIt = rankIt->second.find(dieId);
    if (dieIt == rankIt->second.end()) {
        return {};
    }
    return dieIt->second;
}

std::map<uint16_t, uint16_t> AllRankParamRecorder::GetCKESnapshot(uint32_t rankId, uint32_t dieId) const
{
    auto rankIt = curCKE.find(rankId);
    if (rankIt == curCKE.end()) {
        return {};
    }
    auto dieIt = rankIt->second.find(dieId);
    if (dieIt == rankIt->second.end()) {
        return {};
    }
    return dieIt->second;
}

HcclResult AllRankParamRecorder::CheckAllPostMatch()
{
    for (const auto& rankPair : seenPost) {
        RankId rank = rankPair.first;
        for (const auto& diePair : rankPair.second) {
            uint32_t dieId = diePair.first;
            for (const auto& regPair: diePair.second) {
                uint16_t regId = regPair.first;
                for (const auto *post : regPair.second) {
                    const auto *meta = GetPostNodeMeta(post);
                    if (meta != nullptr) {
                        HCCL_VM_WARN("{} Found CCU post/local-post tasks that were never consumed by "
                            "any Wait task, firstUnconsumedPostNode={}, unconsumedPostCount={}, waitRankId={}, "
                            "dieId={}, "
                            "ckeId={}, recordRankId={}, remainingCkeMask=0x{:x}, isLocal={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED).c_str(),
                            post == nullptr ? "node=null" : post->Describe().c_str(), regPair.second.size(),
                            meta->waitRankId, meta->dieId, meta->ckeId,
                            meta->recordRankId, meta->remainingCkeMask, meta->isLocal);
                        continue;
                    }
                    HCCL_VM_WARN("{} Found CCU post/local-post tasks that were never consumed by any "
                        "Wait task, firstUnconsumedPostNode={}, unconsumedPostCount={}, waitRankId={}, dieId={}, "
                        "ckeId={}", MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED).c_str(),
                        post == nullptr ? "node=null" : post->Describe().c_str(), regPair.second.size(), rank,
                        dieId, regId);
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::SetHBM(uint32_t rankId, uint32_t dieId, uint64_t hbmAddr,
    const std::vector<uint64_t>& data) {
    if (data.size() % 8 != 0) {
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
    data.clear();
    return HCCL_E_PARA;
}

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
