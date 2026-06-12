/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tp_qos.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "log.h"
#include "hccp.h"
#include "exception_util.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

namespace {

constexpr uint32_t kUboeEightTpPolicyCount = 8U;
constexpr uint8_t kUboeDefaultDscp = 33U;

// GetTpInfo 写 SL/DSCP 需在返回前完成；HrtRaSetTpAttrAsync 内部会 WaitRequestResult 阻塞轮询，
// 调用返回时 SetTpAttr 已完成，故此处为同步语义（非可重复 poll 的 RaSetTpAttrAsync）。
static HcclResult SetTpAttrSync(RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t attrBitmap,
    struct TpAttr &attr, const char *logTag)
{
    RequestHandle reqHandle = 0;
    HcclResult hret = HcclResult::HCCL_SUCCESS;
    TRY_CATCH_RETURN(hret = HrtRaSetTpAttrAsync(ctxHandle, tpHandle, attrBitmap, attr, reqHandle));
    if (hret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[%s] HrtRaSetTpAttrAsync failed hcclRet[%d] tpHandle[%llu].", logTag,
            static_cast<int>(hret), tpHandle);
    }
    return hret;
}

static uint32_t ResolveSlAvailableCntForPolicy(uint16_t slMask, uint32_t slLevelCount)
{
    uint32_t slAvailableCnt = TpQosCalSlAvailableCnt(slMask);
    if (slLevelCount != 0U) {
        slAvailableCnt = std::min(slLevelCount, slAvailableCnt);
    }
    return slAvailableCnt;
}

static uint32_t MapUboeEightTpSlFromMask(uint32_t qos, uint16_t slMask, uint32_t slAvailableCnt)
{
    const uint32_t q = qos & 7U;
    if (slAvailableCnt == 0U) {
        return 0U;
    }
    if (slAvailableCnt == 1U) {
        return TpQosSlValueAtRankInMask16(slMask, 0U);
    }
    if (slAvailableCnt == 2U) {
        const uint32_t slRank = (q >= 4U) ? 0U : 1U;
        return TpQosSlValueAtRankInMask16(slMask, slRank);
    }
    uint32_t slRank = 0U;
    if (q >= 5U) {
        slRank = 0U;
    } else if (q >= 3U) {
        slRank = (slAvailableCnt - 1U) / 2U;
    } else {
        slRank = slAvailableCnt - 1U;
    }
    if (slRank >= slAvailableCnt) {
        slRank = slAvailableCnt - 1U;
    }
    return TpQosSlValueAtRankInMask16(slMask, slRank);
}

static bool ApplyUbcQosTpSlPolicyGrouped(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t slRawCnt, uint32_t slAvailableCnt, uint32_t &tpListIndexOut, uint32_t &mappedSlOut,
    const char *logTag);

static bool TryApplyUboeEightTpQosPolicy(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag)
{
    if (policy.tpProtocol != TpProtocol::UBOE || policy.loopFirstTpLowestSl) {
        return false;
    }
    const uint32_t slAvailableCnt = ResolveSlAvailableCntForPolicy(slMask, policy.slLevelCount);
    if (nTp != kUboeEightTpPolicyCount || slAvailableCnt == 0U) {
        return false;
    }
    const uint32_t qos = policy.qos & 7U;
    static constexpr uint8_t kUboeEightTpIndexByQos[8] = {7U, 6U, 5U, 4U, 3U, 2U, 1U, 0U};
    tpListIndexOut = kUboeEightTpIndexByQos[qos];
    mappedSlOut = MapUboeEightTpSlFromMask(qos, slMask, slAvailableCnt);
    HCCL_INFO("[%s][TryApplyUboeEightTpQosPolicy] qos[%u] tpListIndex[%u] mappedSl[%u] slMask[0x%x] "
              "slAvailableCnt[%u] tpProtocol[%s].",
        logTag, qos, tpListIndexOut, mappedSlOut, static_cast<unsigned>(slMask), slAvailableCnt,
        policy.tpProtocol.Describe().c_str());
    return true;
}

static bool ApplyLoopFirstTpLowestSl(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t slRawCnt, uint32_t slAvailableCnt, uint32_t &tpListIndexOut, uint32_t &mappedSlOut,
    const char *logTag)
{
    (void)policy;
    tpListIndexOut = 0;
    mappedSlOut = TpQosSlValueAtRankInMask16(slMask, 0);
    HCCL_INFO("[%s][ApplyUbcQosTpSlPolicy] loopFirstTpLowestSl: nTp[%u] slRawCnt[%u] slAvailableCnt[%u] "
              "slMask[0x%x] tpListIdx[0] mappedSl[%u].",
        logTag, nTp, slRawCnt, slAvailableCnt, static_cast<unsigned>(slMask),
        static_cast<unsigned>(mappedSlOut & 0xFU));
    return true;
}

static bool ApplyUbcQosTpSlPolicyGrouped(const TpQosPolicyInput &policy, const uint32_t nTp, const uint16_t slMask,
    const uint32_t slRawCnt, const uint32_t slAvailableCnt, uint32_t &tpListIndexOut, uint32_t &mappedSlOut,
    const char *logTag)
{
    if (nTp == 0U || slAvailableCnt == 0U) {
        HCCL_WARNING("[%s][ApplyUbcQosTpSlPolicy] nTp or slAvailableCnt zero: nTp[%u] slAvailableCnt[%u] "
                     "slMask[0x%x].",
            logTag, nTp, slAvailableCnt, static_cast<unsigned>(slMask));
        return false;
    }
    const uint32_t k = std::min(nTp, slAvailableCnt);
    if (k == 0U) {
        return false;
    }
    const uint32_t numGroups = std::min(8U, k);
    const uint32_t qos = policy.qos & 7U;
    const uint32_t groupIdx =
        (k == 3U) ? (qos < 3U ? 0U : (qos < 5U ? 1U : 2U)) : ((qos * numGroups) / 8U);
    const uint32_t slotIdx = (groupIdx * k) / numGroups;
    if (slotIdx >= k || slotIdx >= nTp) {
        HCCL_WARNING("[%s][ApplyUbcQosTpSlPolicy] slotIdx out of range: nTp[%u] slRawCnt[%u] slAvailableCnt[%u] "
                     "k[%u] numGroups[%u] qos[%u] groupIdx[%u] slotIdx[%u] slMask[0x%x].",
            logTag, nTp, slRawCnt, slAvailableCnt, k, numGroups, qos, groupIdx, slotIdx,
            static_cast<unsigned>(slMask));
        return false;
    }
    const uint32_t slRank = (slAvailableCnt - 1U) - slotIdx;
    if (slRank >= slAvailableCnt) {
        HCCL_WARNING("[%s][ApplyUbcQosTpSlPolicy] slRank out of range: nTp[%u] slAvailableCnt[%u] k[%u] slRank[%u] "
                     "slMask[0x%x].",
            logTag, nTp, slAvailableCnt, k, slRank, static_cast<unsigned>(slMask));
        return false;
    }
    tpListIndexOut = (k - 1U) - slotIdx;
    mappedSlOut = TpQosSlValueAtRankInMask16(slMask, slRank);
    return true;
}

static bool ApplyUbcQosTpSlPolicy(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag)
{
    const uint32_t slRawCnt = TpQosCalSlAvailableCnt(slMask);
    uint32_t slAvailableCnt = slRawCnt;
    if (slAvailableCnt == 0U) {
        HCCL_WARNING("[%s][ApplyUbcQosTpSlPolicy] slMask empty: nTp[%u] slMask[0x%x].", logTag, nTp,
            static_cast<unsigned>(slMask));
        return false;
    }
    if (policy.slLevelCount != 0U) {
        slAvailableCnt = std::min(policy.slLevelCount, slAvailableCnt);
    }
    if (policy.loopFirstTpLowestSl) {
        return ApplyLoopFirstTpLowestSl(policy, nTp, slMask, slRawCnt, slAvailableCnt, tpListIndexOut, mappedSlOut,
            logTag);
    }
    return ApplyUbcQosTpSlPolicyGrouped(policy, nTp, slMask, slRawCnt, slAvailableCnt, tpListIndexOut, mappedSlOut,
        logTag);
}

static bool ApplyTpQosSlPolicyInternal(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag)
{
    if (TryApplyUboeEightTpQosPolicy(policy, nTp, slMask, tpListIndexOut, mappedSlOut, logTag)) {
        return true;
    }
    return ApplyUbcQosTpSlPolicy(policy, nTp, slMask, tpListIndexOut, mappedSlOut, logTag);
}

static uint32_t ResolveUbcGroupFirstHcclQos(uint32_t qos, uint32_t nTp, uint32_t slAvailableCnt)
{
    const uint32_t q = qos & 7U;
    if (nTp == 0U || slAvailableCnt == 0U) {
        return q;
    }
    const uint32_t k = std::min(nTp, slAvailableCnt);
    const uint32_t numGroups = std::min(8U, k);
    const uint32_t groupIdx =
        (k == 3U) ? (q < 3U ? 0U : (q < 5U ? 1U : 2U)) : ((q * numGroups) / 8U);
    if (k == 3U) {
        static constexpr uint8_t kUboeGroupFirstQos[3] = {0U, 3U, 5U};
        return (groupIdx < 3U) ? static_cast<uint32_t>(kUboeGroupFirstQos[groupIdx]) : 0U;
    }
    for (uint32_t candidate = 0U; candidate <= 7U; ++candidate) {
        if (((candidate * numGroups) / 8U) == groupIdx) {
            return candidate;
        }
    }
    return q;
}

static bool ParseDscpFromCfgByQos(const std::string &cfg, uint8_t qos, uint8_t &dscpOut)
{
    std::vector<uint32_t> nums;
    nums.reserve(32);
    uint32_t cur = 0;
    bool inNum = false;
    for (char ch : cfg) {
        if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
            cur = cur * 10U + static_cast<uint32_t>(ch - '0');
            inNum = true;
            continue;
        }
        if (inNum) {
            nums.push_back(cur);
            cur = 0;
            inNum = false;
        }
    }
    if (inNum) {
        nums.push_back(cur);
    }
    if (nums.empty()) {
        return false;
    }
    if (nums.size() > static_cast<size_t>(qos)) {
        const uint32_t dscp = nums[qos];
        if (dscp <= 63U) {
            dscpOut = static_cast<uint8_t>(dscp);
            return true;
        }
    }
    for (size_t i = 0; i + 1 < nums.size(); i += 2) {
        if (nums[i] == qos && nums[i + 1] <= 63U) {
            dscpOut = static_cast<uint8_t>(nums[i + 1]);
            return true;
        }
    }
    return false;
}

} // namespace

uint32_t TpQosCalSlAvailableCnt(uint32_t mask)
{
    uint32_t c = 0;
    for (uint32_t i = 0; i < 16U; ++i) {
        if ((mask & (1U << i)) != 0U) {
            ++c;
        }
    }
    return c;
}

uint32_t TpQosSlValueAtRankInMask16(uint32_t mask, uint32_t rank)
{
    uint32_t seen = 0;
    for (uint32_t bit = 0; bit < 16U; ++bit) {
        if ((mask & (1U << bit)) != 0U) {
            if (seen == rank) {
                return bit;
            }
            ++seen;
        }
    }
    return 0;
}

uint16_t TpQosReadSlAvailableMask16(const struct TpAttr &attr)
{
    return static_cast<uint16_t>(attr.slBitmap);
}

bool TpQosApplySlPolicy(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag)
{
    return ApplyTpQosSlPolicyInternal(policy, nTp, slMask, tpListIndexOut, mappedSlOut, logTag);
}

HcclResult TpQosSelectTpListEntry(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag)
{
    const uint32_t slAvailableCnt = TpQosCalSlAvailableCnt(slMask);
    if (slAvailableCnt == 0U) {
        HCCL_ERROR("[%s] sl_available mask empty, nTp[%u] slMask[0x%x] tpProtocol[%s].", logTag, nTp,
            static_cast<unsigned>(slMask), policy.tpProtocol.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (!TpQosApplySlPolicy(policy, nTp, slMask, tpListIndexOut, mappedSlOut, logTag)) {
        HCCL_ERROR("[%s] ApplySlPolicy failed, nTp[%u] slAvailableCnt[%u] slMask[0x%x] qos[%u] tpProtocol[%s].",
            logTag, nTp, slAvailableCnt, static_cast<unsigned>(slMask), policy.qos & 0xFFU,
            policy.tpProtocol.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (tpListIndexOut >= nTp) {
        HCCL_ERROR("[%s] tpListIndex out of range: tpListIndex[%u] nTp[%u] mappedSl[%u].", logTag, tpListIndexOut,
            nTp, static_cast<unsigned>(mappedSlOut & 0xFU));
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

uint8_t TpQosResolveUboeDscpLookupQos(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask)
{
    const uint8_t requestQos = static_cast<uint8_t>(policy.qos & 0xFFU);
    uint32_t dummyTpIdx = 0U;
    uint32_t dummySl = 0U;
    if (TryApplyUboeEightTpQosPolicy(policy, nTp, slMask, dummyTpIdx, dummySl, "TpQos")) {
        return requestQos;
    }
    if (policy.loopFirstTpLowestSl) {
        return 0U;
    }
    uint32_t slAvailableCnt = TpQosCalSlAvailableCnt(slMask);
    if (policy.slLevelCount != 0U) {
        slAvailableCnt = std::min(policy.slLevelCount, slAvailableCnt);
    }
    return static_cast<uint8_t>(ResolveUbcGroupFirstHcclQos(policy.qos, nTp, slAvailableCnt));
}

bool TpQosGetDscpByQosFromHccnCfg(uint32_t devPhyId, uint8_t qos, uint8_t &dscpOut)
{
    struct RaInfo info {};
    info.mode = NETWORK_OFFLINE;
    info.phyId = devPhyId;
    constexpr unsigned int kCfgBufLen = 2048U;
    std::vector<char> value(kCfgBufLen, 0);
    unsigned int valueLen = kCfgBufLen;
    const int ret = RaGetHccnCfg(&info, HCCN_CFG_QOS_DSCP, value.data(), &valueLen);
    if (ret != 0 || valueLen == 0U) {
        HCCL_WARNING("[TpQos][GetDscpByQosFromHccnCfg] RaGetHccnCfg failed, ret[%d] valueLen[%u] devPhyId[%u] "
                     "qos[%u].",
            ret, valueLen, devPhyId, static_cast<unsigned>(qos));
        return false;
    }
    if (valueLen > kCfgBufLen) {
        valueLen = kCfgBufLen;
    }
    const std::string cfg(value.data(), valueLen);
    return ParseDscpFromCfgByQos(cfg, qos, dscpOut);
}

HcclResult TpQosCommitMappedSlToTpAttr(RdmaHandle rdmaHandle, uint64_t tpHandle, uint32_t mappedSl,
    const char *logTag)
{
    if (tpHandle == 0U) {
        HCCL_ERROR("[%s][CommitMappedSlToTpAttr] tpHandle is 0", logTag);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (!rdmaHandle) {
        HCCL_ERROR("[%s][CommitMappedSlToTpAttr] rdmaHandle is null tpHandle[%llu]", logTag, tpHandle);
        return HcclResult::HCCL_E_INTERNAL;
    }
    struct TpAttr tpSlAttr {};
    tpSlAttr.sl = static_cast<uint8_t>(mappedSl & 0xFU);
    return SetTpAttrSync(rdmaHandle, tpHandle, kTpQosAttrBitmapSl, tpSlAttr, logTag);
}

HcclResult TpQosCommitUboeDscpToTpAttr(RdmaHandle rdmaHandle, uint64_t tpHandle, uint8_t dscp, const char *logTag)
{
    if (tpHandle == 0U || !rdmaHandle) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    struct TpAttr tpDscpAttr {};
    tpDscpAttr.dscp = static_cast<uint8_t>(dscp & 0x3FU);
    return SetTpAttrSync(rdmaHandle, tpHandle, kTpQosAttrBitmapDscp, tpDscpAttr, logTag);
}

HcclResult TpQosCommitAttrsAfterSlMapping(RdmaHandle rdmaHandle, bool isPcieStd, const TpQosPolicyInput &policy,
    const struct TpAttr &tpAttr, uint64_t tpHandle, uint32_t mappedSl, uint32_t nTp, uint16_t slMask,
    uint32_t devPhyId, const char *logTag)
{
    if (isPcieStd) {
        HCCL_INFO("[%s] pcie std mainboard: skip SetTpAttr, devPhyId[%u] tpProtocol[%s] tpHandle[%llu].", logTag,
            devPhyId, policy.tpProtocol.Describe().c_str(), tpHandle);
        return HcclResult::HCCL_SUCCESS;
    }
    if (TpQosProtocolCommitsMappedSl(policy.tpProtocol)) {
        CHK_RET(TpQosCommitMappedSlToTpAttr(rdmaHandle, tpHandle, mappedSl, logTag));
    }
    if (policy.tpProtocol == TpProtocol::UBOE && tpAttr.dscpConfigMode == 0) {
        const uint8_t dscpBefore = static_cast<uint8_t>(tpAttr.dscp & 0x3FU);
        const uint8_t requestQos = static_cast<uint8_t>(policy.qos & 0xFFU);
        const uint8_t dscpLookupQos = TpQosResolveUboeDscpLookupQos(policy, nTp, slMask);
        uint8_t dscp = kUboeDefaultDscp;
        if (!TpQosGetDscpByQosFromHccnCfg(devPhyId, dscpLookupQos, dscp)) {
            HCCL_WARNING("[%s] UBOE dscp: read HCCN_CFG_QOS_DSCP failed, use default dscp[%u] devPhyId[%u] "
                         "dscpLookupQos[%u] tpHandle[%llu].",
                logTag, static_cast<unsigned>(kUboeDefaultDscp), devPhyId, static_cast<unsigned>(dscpLookupQos),
                tpHandle);
            dscp = kUboeDefaultDscp;
        }
        CHK_RET(TpQosCommitUboeDscpToTpAttr(rdmaHandle, tpHandle, dscp, logTag));
        HCCL_INFO("[%s] UBOE dscp updated: tpHandle[%llu] requestQos[%u] dscpLookupQos[%u] dscpBefore[%u] "
                  "dscpAfter[%u].",
            logTag, tpHandle, static_cast<unsigned>(requestQos), static_cast<unsigned>(dscpLookupQos),
            static_cast<unsigned>(dscpBefore), static_cast<unsigned>(dscp));
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
