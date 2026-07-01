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
#include <arpa/inet.h>
#include <cctype>
#include <string>
#include <vector>

#include "log.h"
#include "securec.h"
#include "hccp.h"
#include "hccp_ctx.h"
#include "exception_util.h"
#include "orion_adapter_hccp.h"
#include "rdma_handle_manager.h"

namespace Hccl {

namespace {

constexpr uint8_t kUboeDefaultDscp = 33U;

static HcclResult Ipv4ToIpArray(const char *ipv4Str, uint8_t ipArr[16])
{
    struct in_addr addr {};
    if (ipv4Str == nullptr || inet_pton(AF_INET, ipv4Str, &addr) != 1) {
        return HcclResult::HCCL_E_PARA;
    }
    if (memset_s(ipArr, 16U, 0, 16U) != EOK) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    const uint32_t ipNet = addr.s_addr;
    ipArr[12] = static_cast<uint8_t>(ipNet & 0xFFU);
    ipArr[13] = static_cast<uint8_t>((ipNet >> 8U) & 0xFFU);
    ipArr[14] = static_cast<uint8_t>((ipNet >> 16U) & 0xFFU);
    ipArr[15] = static_cast<uint8_t>((ipNet >> 24U) & 0xFFU);
    return HcclResult::HCCL_SUCCESS;
}

// GetTpInfo 写 SL/DSCP 需在返回前完成。
// 异步路径：HrtRaSetTpAttrAsync 内部已 WaitRequestResult，返回时 SetTpAttr 已完成。
static HcclResult SetTpAttrAsync(RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t attrBitmap,
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

static HcclResult SetTpAttrSync(RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t attrBitmap,
    struct TpAttr &attr, const char *logTag)
{
    const s32 ret = RaCtxSetTpAttr(ctxHandle, tpHandle, attrBitmap, &attr);
    if (ret != 0) {
        HCCL_ERROR("[%s] RaCtxSetTpAttr failed ret[%d] tpHandle[%llu] attrBitmap[0x%x].", logTag, ret,
            tpHandle, attrBitmap);
        return HcclResult::HCCL_E_NETWORK;
    }
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult SetTpAttrByPath(bool isSync, RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t attrBitmap,
    struct TpAttr &attr, const char *logTag)
{
    if (isSync) {
        return SetTpAttrSync(ctxHandle, tpHandle, attrBitmap, attr, logTag);
    }
    return SetTpAttrAsync(ctxHandle, tpHandle, attrBitmap, attr, logTag);
}

static HcclResult CommitMappedSlToTpAttr(bool isSync, RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t mappedSl,
    const char *logTag)
{
    if (tpHandle == 0U || !ctxHandle) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    struct TpAttr tpSlAttr {};
    tpSlAttr.sl = static_cast<uint8_t>(mappedSl & 0xFU);
    return SetTpAttrByPath(isSync, ctxHandle, tpHandle, kTpQosAttrBitmapSl, tpSlAttr, logTag);
}

static HcclResult CommitUboeNetAttrsToTpAttr(bool isSync, RdmaHandle rdmaHandle, uint64_t tpHandle,
    const TpAttr &tpAttr, const IpAddress &locIpv4Addr, const IpAddress &rmtIpv4Addr, bool setDscp, uint8_t dscp,
    const char *logTag)
{
    if (tpHandle == 0U || !rdmaHandle) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    struct TpAttr netAttr = tpAttr;
    const std::string localIp = locIpv4Addr.GetIpStr();
    const std::string rmtIp = rmtIpv4Addr.GetIpStr();
    CHK_RET(Ipv4ToIpArray(localIp.c_str(), netAttr.sip));
    CHK_RET(Ipv4ToIpArray(rmtIp.c_str(), netAttr.dip));
    if (setDscp) {
        netAttr.dscp = static_cast<uint8_t>(dscp & 0x3FU);
    }
    HCCL_INFO("[%s][CommitUboeNetAttrsToTpAttr] tpHandle[%llu] localIpv4[%s] rmtIpv4[%s] setDscp[%d] "
              "dscp[%u] attrBitmap[0x%x].",
        logTag, tpHandle, localIp.c_str(), rmtIp.c_str(), static_cast<int>(setDscp),
        static_cast<unsigned>(netAttr.dscp & 0x3FU), kTpQosAttrBitmapUboeNetWithDscp);
    return SetTpAttrByPath(isSync, rdmaHandle, tpHandle, kTpQosAttrBitmapUboeNetWithDscp, netAttr, logTag);
}

static bool ApplyLoopFirstTpLowestSl(const TpQosPolicyInput &policy, uint16_t slMask,
    uint32_t slAvailableCnt, uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag)
{
    (void)policy;
    tpListIndexOut = 0U;
    mappedSlOut = TpQosSlValueAtRankInMask16(slMask, 0U);
    HCCL_INFO("[%s][ApplyQosTpSlPolicy] loopFirstTpLowestSl: slAvailableCnt[%u] "
              "slMask[0x%x] tpListIdx[0] mappedSl[%u].",
        logTag, slAvailableCnt, static_cast<unsigned>(slMask),
        static_cast<unsigned>(mappedSlOut & 0xFU));
    return true;
}

static bool ParseDscpFromCfgByQos(const std::string &cfg, uint8_t qos, uint8_t &dscpOut)
{
    constexpr size_t initialReserveSize = 32;
    std::vector<uint32_t> nums;
    nums.reserve(initialReserveSize);
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
    static constexpr size_t pairStep = 2;
    for (size_t i = 0; i + 1 < nums.size(); i += pairStep) {
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

bool TpQosApplySlPolicy(const TpQosPolicyInput &policy, uint16_t slMask,
    uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag)
{
    const uint32_t slAvailableCnt = TpQosCalSlAvailableCnt(slMask);
    if (slAvailableCnt == 0U) {
        return false;
    }
    if (policy.loopFirstTpLowestSl) {
        return ApplyLoopFirstTpLowestSl(policy, slMask, slAvailableCnt, tpListIndexOut, mappedSlOut, logTag);
    }

    const uint32_t qos = policy.qos & 7U;
    const uint32_t numGroups = slAvailableCnt;
    const uint32_t groupIdx =
        (numGroups == 3U) ? (qos < 3U ? 0U : (qos < 5U ? 1U : 2U)) : ((qos * numGroups) / 8U);
    if (groupIdx >= numGroups) {
        return false;
    }

    tpListIndexOut = 0U;
    const uint32_t slRank = (slAvailableCnt - 1U) - groupIdx;
    if (slRank >= slAvailableCnt) {
        return false;
    }
    mappedSlOut = TpQosSlValueAtRankInMask16(slMask, slRank);
    return true;
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
    if (!TpQosApplySlPolicy(policy, slMask, tpListIndexOut, mappedSlOut, logTag)) {
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
    (void)nTp;
    (void)slMask;
    if (policy.loopFirstTpLowestSl) {
        return 0U;
    }
    return static_cast<uint8_t>(policy.qos & 0xFFU);
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

uint32_t TpQosBuildBootstrapAttrBitmap(TpProtocol tpProtocol)
{
    uint32_t bitmap = (1U << kTpQosAttrSlAvailableBit) | kTpQosAttrBitmapSl;
    if (tpProtocol == TpProtocol::UBOE) {
        bitmap |= kTpQosAttrBitmapDscp | (1U << kTpQosAttrDscpConfigModeBit);
    }
    return bitmap;
}

RdmaHandle TpQosResolveUbRdmaHandle(const bool isSync, const uint32_t devPhyId, const IpAddress &locAddr)
{
    if (isSync) {
        IpAddress addr = locAddr;
        return RdmaHandleManager::GetInstance().GetByAddr(devPhyId, LinkProtoType::UB, addr,
            PortDeploymentType::HOST_NET);
    }
    return RdmaHandleManager::GetInstance().GetByIp(devPhyId, locAddr);
}

HcclResult TpQosSyncGetTpAttr(RdmaHandle rdmaHandle, const uint64_t tpHandle, const TpProtocol tpProtocol,
    struct TpAttr &tpAttr, uint32_t &attrBitmap, const char *logTag)
{
    (void)memset_s(&tpAttr, sizeof(tpAttr), 0, sizeof(tpAttr));
    attrBitmap = TpQosBuildBootstrapAttrBitmap(tpProtocol);
    if (!rdmaHandle) {
        HCCL_ERROR("[%s][SyncGetTpAttr] rdmaHandle is null tpHandle[%llu].", logTag, tpHandle);
        return HcclResult::HCCL_E_INTERNAL;
    }
    const s32 ret = RaCtxGetTpAttr(rdmaHandle, tpHandle, &attrBitmap, &tpAttr);
    if (ret != 0) {
        HCCL_ERROR("[%s][SyncGetTpAttr] RaCtxGetTpAttr failed ret[%d] tpHandle[%llu].", logTag, ret, tpHandle);
        return HcclResult::HCCL_E_NETWORK;
    }
    HCCL_INFO("[%s][SyncGetTpAttr] RaCtxGetTpAttr ok, tpHandle[%llu] attrBitmap[0x%x].", logTag, tpHandle, attrBitmap);
    return HcclResult::HCCL_SUCCESS;
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
    return CommitMappedSlToTpAttr(false, rdmaHandle, tpHandle, mappedSl, logTag);
}

HcclResult TpQosCommitAttrsAfterSlMapping(RdmaHandle rdmaHandle, bool isPcieStd, const TpQosPolicyInput &policy,
    const struct TpAttr &tpAttr, uint64_t tpHandle, uint32_t mappedSl, uint32_t nTp, uint16_t slMask,
    uint32_t devPhyId, const char *logTag, const bool isSync)
{
    if (isPcieStd) {
        HCCL_INFO("[%s] pcie std mainboard: skip SetTpAttr, devPhyId[%u] tpProtocol[%s] tpHandle[%llu].", logTag,
            devPhyId, policy.tpProtocol.Describe().c_str(), tpHandle);
        return HcclResult::HCCL_SUCCESS;
    }
    if (TpQosProtocolCommitsMappedSl(policy.tpProtocol)) {
        CHK_RET(CommitMappedSlToTpAttr(isSync, rdmaHandle, tpHandle, mappedSl, logTag));
    }
    if (policy.tpProtocol == TpProtocol::UBOE) {
        if (tpAttr.dscpConfigMode == 1) {
            CHK_RET(CommitUboeNetAttrsToTpAttr(isSync, rdmaHandle, tpHandle, tpAttr, policy.locIpv4Addr,
                policy.rmtIpv4Addr, false, 0U, logTag));
            return HcclResult::HCCL_SUCCESS;
        }
        const uint8_t dscpBefore = static_cast<uint8_t>(tpAttr.dscp & 0x3FU);
        const uint8_t requestQos = static_cast<uint8_t>(policy.qos & 0xFFU);
        const uint8_t dscpLookupQos = TpQosResolveUboeDscpLookupQos(policy, nTp, slMask);
        uint8_t dscp = kUboeDefaultDscp;
        (void)TpQosGetDscpByQosFromHccnCfg(devPhyId, dscpLookupQos, dscp);
        CHK_RET(CommitUboeNetAttrsToTpAttr(isSync, rdmaHandle, tpHandle, tpAttr, policy.locIpv4Addr,
            policy.rmtIpv4Addr, true, dscp, logTag));
        HCCL_INFO("[%s] UBOE net attrs updated: tpHandle[%llu] requestQos[%u] dscpLookupQos[%u] dscpBefore[%u] "
                  "dscpAfter[%u].",
            logTag, tpHandle, static_cast<unsigned>(requestQos), static_cast<unsigned>(dscpLookupQos),
            static_cast<unsigned>(dscpBefore), static_cast<unsigned>(dscp));
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
