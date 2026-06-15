/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tp_manager.h"

#include <algorithm>
#include <cctype>
#include <string>

#include "exception_util.h"
#include "hccl_common_v2.h"
#include "invalid_params_exception.h"
#include "env_config/env_config.h"
#include "network_api_exception.h"
#include "tp_qos.h"

#include "hccp.h"
#include "hccp_async_ctx.h"
#include "hccp_ctx.h"
#include "dev_type.h"
#include "orion_adapter_rts.h"
#include "rdma_handle_manager.h"

namespace Hccl {

namespace {

constexpr uint32_t kGetTpAttrOpcode = 106U;
constexpr uint32_t kGetTpAttrVersion = 2U;

static constexpr uint32_t kPcieStdMappedSl = kTpQosPcieStdMappedSl;

static HcclResult IsPcieStdMainboard(uint32_t devLogicId, bool &isPcieStd)
{
    isPcieStd = false;
    HcclMainboardId mainboardId = HcclMainboardId::MAINBOARD_OTHERS;
    CHK_RET(HrtGetMainboardId(devLogicId, mainboardId));
    isPcieStd = (mainboardId == HcclMainboardId::MAINBOARD_PCIE_STD);
    return HcclResult::HCCL_SUCCESS;
}

static bool DeviceSupportsRaGetTpAttr(uint32_t phyId)
{
    u32 tpAttrVersion = 0;
    const s32 ret = RaGetInterfaceVersion(phyId, kGetTpAttrOpcode, &tpAttrVersion);
    return (ret == 0 && tpAttrVersion >= kGetTpAttrVersion);
}

} // namespace

TpManager& TpManager::GetInstance(const int32_t deviceLogicId)
{
    static TpManager tpManager[MAX_MODULE_DEVICE_NUM + 1];

    if (deviceLogicId < 0 ||
        static_cast<uint32_t>(deviceLogicId) > MAX_MODULE_DEVICE_NUM) {
        THROW<InvalidParamsException>("[TpManager][%s] failed to get instance, "
            "devLogicId[%d] should be less than %u.", __func__,
            deviceLogicId, MAX_MODULE_DEVICE_NUM);
    }

    tpManager[deviceLogicId].devLogicId = deviceLogicId;

    return tpManager[deviceLogicId];
}

void TpManager::Init()
{
    if (initFlag) {
        return;
    }

    devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);
    initFlag = true;
}

void TpManager::SetIsHost()
{
    isHost = true;
}

bool TpManager::CheckRequestResult(RequestHandle &reqHandle) const
{
    if (reqHandle == 0) {
        return true;
    }

    ReqHandleResult result = HrtRaGetAsyncReqResult(reqHandle);
    if (result == ReqHandleResult::NOT_COMPLETED) {
        return false;
    }

    if (result != ReqHandleResult::COMPLETED) {
        THROW<InternalException>("[TpManager][%s] failed, result[%s] is unexpected.",
            __func__, result.Describe().c_str());
    }

    return true;
}

HcclResult CheckTpProtocol(const TpProtocol tpProtocol) {
    if (tpProtocol != TpProtocol::CTP && tpProtocol != TpProtocol::TP && tpProtocol != TpProtocol::UBOE) {
        HCCL_WARNING("[TpManager][%s] failed, tpProtocol[%d] is not supported.",
            __func__, tpProtocol);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::FinishGetTpInfoFromReq(ReqQosMap &qosReqMap, const ReqQosMap::iterator it,
    std::unique_lock<std::mutex> &reqCtxLock, const RaUbGetTpInfoParam &param, TpInfo &tpInfo, const bool withSlPolicy)
{
    RequestCtx completedReqCtx = std::move(it->second);
    qosReqMap.erase(it);
    const HcclResult ret = HandleCompletedRequest(std::move(completedReqCtx), param, tpInfo, withSlPolicy);
    reqCtxLock.unlock();
    return ret;
}

HcclResult TpManager::AdvanceDeviceWaitListPhase(const RaUbGetTpInfoParam &param, RequestCtx &reqCtx,
    ReqQosMap &qosReqMap, const ReqQosMap::iterator it, std::unique_lock<std::mutex> &reqCtxLock, TpInfo &tpInfo)
{
    if (reqCtx.tpInfoNum == 0U) {
        qosReqMap.erase(it);
        reqCtxLock.unlock();
        HCCL_WARNING("[TpManager][%s] failed to find tp info, tpInfoNum is 0, param[%s].", __func__,
            param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboard(devLogicId, isPcieStd));
    if (isPcieStd) {
        const struct HccpTpInfo *list = reinterpret_cast<const struct HccpTpInfo *>(reqCtx.dataBuffer.data());
        HCCL_INFO("[TpManager][%s] pcie std mainboard: skip GetTpAttr, devPhyId[%u] tpInfoNum[%u] mappedSl[%u] "
                  "tpHandle[%llu] param[%s].",
            __func__, devPhyId, reqCtx.tpInfoNum, kPcieStdMappedSl,
            static_cast<unsigned long long>(list[0].tpHandle), param.Describe().c_str());
        return FinishGetTpInfoFromReq(qosReqMap, it, reqCtxLock, param, tpInfo, false);
    }
    if (DeviceSupportsRaGetTpAttr(devPhyId)) {
        const struct HccpTpInfo *list = reinterpret_cast<const struct HccpTpInfo *>(reqCtx.dataBuffer.data());
        HCCL_INFO("[TpManager][GetTpInfo] list stage ok, devPhyId[%u] tpInfoNum[%u] firstTpHandle[%llu] param[%s].",
            devPhyId, reqCtx.tpInfoNum, static_cast<unsigned long long>(list[0].tpHandle), param.Describe().c_str());
        try {
            StartGetTpAttrForFirstTpDevice(param, reqCtx);
        } catch (...) {
            qosReqMap.erase(it);
            throw;
        }
        return HcclResult::HCCL_E_AGAIN;
    }
    return FinishGetTpInfoFromReq(qosReqMap, it, reqCtxLock, param, tpInfo, false);
}

HcclResult TpManager::PollGetTpInfoReqCtx(const RaUbGetTpInfoParam &param, TpInfo &tpInfo, const bool isSync)
{
    std::unique_lock<std::mutex> reqCtxLock(GetReqCtxMutex(param.tpProtocol));
    auto &reqCtxMap = GetReqCtxMap(param.tpProtocol);
    const auto &locAddr = param.locAddr;
    const auto &rmtAddr = param.rmtAddr;
    const QosKey qosKey = TpQosMapKeyFromQos(param.qos);

    auto &rmtReqMap = reqCtxMap[locAddr];
    auto &qosReqMap = rmtReqMap[rmtAddr];
    auto it = qosReqMap.find(qosKey);
    if (it == qosReqMap.end()) {
        HCCL_INFO("[TpManager][%s] get new tpInfo, param[%s].", __func__, param.Describe().c_str());

        RequestCtx &reqCtx = qosReqMap[qosKey];
        reqCtx.isSync = isSync;
        StartGetTpInfoListRequest(param, reqCtx, isSync);
        return HcclResult::HCCL_E_AGAIN;
    }

    RequestCtx &reqCtx = it->second;

    if (!reqCtx.isSync && reqCtx.handle != 0U && !CheckRequestResult(reqCtx.handle)) {
        return HcclResult::HCCL_E_AGAIN;
    }

    if (!isHost) {
        if (reqCtx.phase == RequestCtx::ReqPhase::WAIT_LIST) {
            return AdvanceDeviceWaitListPhase(param, reqCtx, qosReqMap, it, reqCtxLock, tpInfo);
        }
        if (reqCtx.phase == RequestCtx::ReqPhase::WAIT_TP_ATTR) {
            return FinishGetTpInfoFromReq(qosReqMap, it, reqCtxLock, param, tpInfo, true);
        }
    }

    return FinishGetTpInfoFromReq(qosReqMap, it, reqCtxLock, param, tpInfo, false);
}

HcclResult TpManager::GetTpInfo(const RaUbGetTpInfoParam &param, TpInfo &tpInfo, bool isSync)
{
    CHK_RET(CheckTpProtocol(param.tpProtocol));
    if (FindAndGetTpInfo(param, tpInfo) == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }

    return PollGetTpInfoReqCtx(param, tpInfo, isSync);
}

HcclResult TpManager::FindAndGetTpAttr(const TpHandle tpHandle, TpAttrInfo &tpAttrInfo)
{
    std::lock_guard<std::mutex> lock(tpAttrCtxMutex);
    auto attrIter = tpAttrCtxMap.find(tpHandle);
    if (attrIter != tpAttrCtxMap.end()) {
        attrIter->second.useCnt += 1;
        tpAttrInfo = attrIter->second.tpAttrInfo;
        return HcclResult::HCCL_SUCCESS;
    }

    return HcclResult::HCCL_E_NOT_FOUND;
}

HcclResult TpManager::GetTpAttr(const GetTpAttrParam &param, TpAttrInfo &tpAttrInfo, RdmaHandle rdmaHandle)
{
    const TpHandle tpHandle = param.tpHandle;
    if (FindAndGetTpAttr(tpHandle, tpAttrInfo) == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }

    std::unique_lock<std::mutex> reqCtxLock(tpAttrReqMutex);
    auto reqCtxIter = tpAttrReqCtxMap.find(tpHandle);
    if (reqCtxIter == tpAttrReqCtxMap.end()) {
        HCCL_INFO("[TpManager][%s] get new tpAttr, param[%s].", __func__,
            param.Describe().c_str());

        TpAttrRequestCtx &reqCtx = tpAttrReqCtxMap[tpHandle];
        CHK_RET(StartGetTpAttrRequest(param, reqCtx, rdmaHandle));
        return HcclResult::HCCL_E_AGAIN;
    }

    auto &reqCtx = reqCtxIter->second;
    if (!CheckRequestResult(reqCtx.handle)) {
        return HcclResult::HCCL_E_AGAIN;
    }

    TpAttrRequestCtx completedReqCtx = reqCtxIter->second;
    tpAttrReqCtxMap.erase(reqCtxIter);
    reqCtxLock.unlock();

    return HandleCompletedTpAttrRequest(std::move(completedReqCtx), tpHandle, tpAttrInfo);
}

HcclResult TpManager::StartGetTpAttrRequest(const GetTpAttrParam &param,
    TpManager::TpAttrRequestCtx &reqCtx, RdmaHandle rdmaHandle) const
{
    void *raReqHandle = nullptr;
    s32 ret = RaGetTpAttrAsync(rdmaHandle, param.tpHandle,
        const_cast<uint32_t*>(&param.attrBitmap), &reqCtx.tpAttr, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        HCCL_ERROR("[TpManager][%s] failed, call RaGetTpAttrAsync error[%d] raReqHandle[%p], "
            "tpHandle[0x%llx] attrBitmap[0x%x].", __func__, ret, raReqHandle,
            param.tpHandle, param.attrBitmap);
        return HcclResult::HCCL_E_NETWORK;
    }

    reqCtx.handle = reinterpret_cast<RequestHandle>(raReqHandle);
    HCCL_INFO("[TpManager][%s] success, tpHandle[0x%llx] reqHandle[%llu].",
        __func__, param.tpHandle, reqCtx.handle);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::HandleCompletedTpAttrRequest(const TpManager::TpAttrRequestCtx reqCtx,
    const TpHandle tpHandle, TpAttrInfo &tpAttrInfo)
{
    TpAttrInfo tmpTpAttrInfo(reqCtx.tpAttr);

    std::lock_guard<std::mutex> lock(tpAttrCtxMutex);
    tpAttrCtxMap[tpHandle] = {std::move(tmpTpAttrInfo), 1};
    
    tpAttrInfo = tpAttrCtxMap[tpHandle].tpAttrInfo;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::ReleaseTpInfo(const RaUbGetTpInfoParam &param, const TpInfo &tpInfo)
{
    const QosKey qosKey = TpQosMapKeyFromQos(param.qos);
    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto &infoMap = GetInfoCtxMap(param.tpProtocol);
    auto lit = infoMap.find(param.locAddr);
    if (lit == infoMap.end()) {
        HCCL_ERROR("[TpManager][%s] failed, tp info is not found, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    auto rit = lit->second.find(param.rmtAddr);
    if (rit == lit->second.end()) {
        HCCL_ERROR("[TpManager][%s] failed, tp info is not found, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    auto qit = rit->second.find(qosKey);
    if (qit == rit->second.end()) {
        HCCL_ERROR("[TpManager][%s] failed, tp info is not found for qosKey[%u], param[%s].", __func__,
            static_cast<unsigned>(qosKey), param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    if (tpInfo.tpHandle != qit->second.tpInfo.tpHandle) {
        HCCL_ERROR("[TpManager][%s] failed, tp info[%llu] is not expected[%llu].",
            __func__, tpInfo.tpHandle, qit->second.tpInfo.tpHandle);
        return HcclResult::HCCL_E_PARA;
    }

    if (qit->second.useCnt > 1) {
        qit->second.useCnt -= 1;
        return HcclResult::HCCL_SUCCESS;
    }

    rit->second.erase(qit);
    if (rit->second.empty()) {
        lit->second.erase(rit);
    }
    if (lit->second.empty()) {
        infoMap.erase(lit);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::ReleaseTpAttr(const TpHandle tpHandle, const TpAttrInfo &tpAttrInfo)
{
    std::lock_guard<std::mutex> lock(tpAttrCtxMutex);
    auto attrIter = tpAttrCtxMap.find(tpHandle);
    if (attrIter == tpAttrCtxMap.end()) {
        HCCL_ERROR("[TpManager][%s] failed, tp attr is not found, "
            "tpHandle[0x%llx].", __func__, tpHandle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    if (attrIter->second.useCnt > 1) {
        attrIter->second.useCnt -= 1;
        return HcclResult::HCCL_SUCCESS;
    }

    tpAttrCtxMap.erase(attrIter);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::GetTpTotalTimeout(const TpAttrInfo &tpAttrInfo, uint32_t &tpTimeOutMs)
{
    uint8_t rawAtGear = tpAttrInfo.tpAttr.at;
    uint8_t rawRetryTimes = tpAttrInfo.tpAttr.retryTimesInit;

    uint8_t finalAtGear = rawAtGear;
    if (rawAtGear < AT_GEAR_MIN || rawAtGear > AT_GEAR_MAX) {
        finalAtGear = AT_GEAR_DEFAULT;
        HCCL_WARNING("%s Invalid at gear[%u], expect [%u, %u], use default gear[%u].",
            __func__, rawAtGear, AT_GEAR_MIN, AT_GEAR_MAX, finalAtGear);
    }

    uint32_t singleAtTimeoutMs = AT_TIMEOUT_MAP[finalAtGear];
    tpTimeOutMs = singleAtTimeoutMs * static_cast<uint32_t>(rawRetryTimes + 1);

    HCCL_INFO("%s TP timeout calc success: raw_at_gear[%u], final_at_gear[%u], "
        "single_timeout[%ums], retry_times[%u], total_timeout[%ums].",
        __func__, rawAtGear, finalAtGear, singleAtTimeoutMs, rawRetryTimes, tpTimeOutMs);

    return HcclResult::HCCL_SUCCESS;
}

uint32_t TpManager::TaHwValueToMs(uint8_t hwValue)
{
    uint8_t gear = hwValue / 8;
    switch (gear) {
        case TA_GEAR_INDEX_0: return TA_TIMEOUT_MS_GEAR0;
        case TA_GEAR_INDEX_1: return TA_TIMEOUT_MS_GEAR1;
        case TA_GEAR_INDEX_2: return TA_TIMEOUT_MS_GEAR2;
        case TA_GEAR_INDEX_3: return TA_TIMEOUT_MS_GEAR3;
        default: return TA_TIMEOUT_MS_GEAR2;
    }
}

uint8_t TpManager::FindMinTaHwValue(uint32_t tpTotalTimeoutMs)
{
    if (tpTotalTimeoutMs < TA_TIMEOUT_MS_GEAR0) {
        return TA_HW_GEAR0_BASE;
    }
    if (tpTotalTimeoutMs < TA_TIMEOUT_MS_GEAR1) {
        return TA_HW_GEAR1_BASE;
    }
    if (tpTotalTimeoutMs < TA_TIMEOUT_MS_GEAR2) {
        return TA_HW_GEAR2_BASE;
    }
    return TA_HW_GEAR3_BASE;
}

uint8_t TpManager::CalcTaTimeout(const TpAttrInfo &tpAttrInfo)
{
    constexpr uint8_t UB_TIMEOUT_DEFAULT = 8;
    uint8_t envValue = static_cast<uint8_t>(EnvConfig::GetInstance().GetRdmaConfig().GetUbTimeOut());
    uint32_t envTimeoutMs = TaHwValueToMs(envValue);
    
    uint32_t tpTimeOutMs = 0;
    (void)GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    
    uint8_t errTimeout = UB_TIMEOUT_DEFAULT;
    if (envTimeoutMs < tpTimeOutMs) {
        errTimeout = FindMinTaHwValue(tpTimeOutMs);
        HCCL_WARNING("[TpManager][%s] Env timeout [%ums] < TP timeout [%ums]. Auto upgrade TA to hw_val[%u] (%ums).",
            __func__, envTimeoutMs, tpTimeOutMs, errTimeout, TaHwValueToMs(errTimeout));
    } else {
        errTimeout = envValue;
        HCCL_INFO("[TpManager][%s] Env timeout [%ums] >= TP timeout [%ums]. Use env gear base hw_val[%u] (%ums).",
            __func__, envTimeoutMs, tpTimeOutMs, envValue, envTimeoutMs);
    }
    
    return errTimeout;
}

HcclResult TpManager::FindAndGetTpInfo(const RaUbGetTpInfoParam &param, TpInfo &tpInfo)
{
    const QosKey qosKey = TpQosMapKeyFromQos(param.qos);
    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto &infoMap = GetInfoCtxMap(param.tpProtocol);
    auto lit = infoMap.find(param.locAddr);
    if (lit == infoMap.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    auto rit = lit->second.find(param.rmtAddr);
    if (rit == lit->second.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    auto qit = rit->second.find(qosKey);
    if (qit == rit->second.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    qit->second.useCnt += 1;
    tpInfo = qit->second.tpInfo;
    return HcclResult::HCCL_SUCCESS;
}

void TpManager::StartGetTpInfoListRequest(const RaUbGetTpInfoParam &param,
    TpManager::RequestCtx &reqCtx, bool isSync) const
{
    reqCtx.phase = RequestCtx::ReqPhase::WAIT_LIST;
    (void)memset_s(&reqCtx.tpAttr, sizeof(reqCtx.tpAttr), 0, sizeof(reqCtx.tpAttr));
    reqCtx.tpAttrBitmap = 0;

    Hccl::IpAddress localIp = param.locAddr;
    RdmaHandle rdmaHandle = isSync
        ? RdmaHandleManager::GetInstance().GetByAddr(devPhyId, LinkProtoType::UB, localIp,
              Hccl::PortDeploymentType::HOST_NET)
        : RdmaHandleManager::GetInstance().GetByIp(devPhyId, param.locAddr);
    if (!rdmaHandle) {
        THROW<InternalException>("[TpManager][%s] can not find rdmaHandle, "
            "devPhyId[%u] locAddr[%s].", __func__, devPhyId,
            param.locAddr.Describe().c_str());
    }
    if (isSync) {
        RaUbGetTpInfo(rdmaHandle, param, reqCtx.dataBuffer, reqCtx.tpInfoNum);
        return;
    }
    reqCtx.handle = RaUbGetTpInfoAsync(rdmaHandle, param, reqCtx.dataBuffer, reqCtx.tpInfoNum);
}

void TpManager::StartGetTpAttrForFirstTpDevice(const RaUbGetTpInfoParam &param, RequestCtx &reqCtx) const
{
    (void)memset_s(&reqCtx.tpAttr, sizeof(reqCtx.tpAttr), 0, sizeof(reqCtx.tpAttr));
    reqCtx.tpAttrBitmap = (1U << kTpQosAttrSlAvailableBit) | kTpQosAttrBitmapSl;
    if (param.tpProtocol == TpProtocol::UBOE) {
        reqCtx.tpAttrBitmap |= kTpQosAttrBitmapDscp | (1U << kTpQosAttrDscpConfigModeBit);
    }
    const struct HccpTpInfo *list = reinterpret_cast<const struct HccpTpInfo *>(reqCtx.dataBuffer.data());
    const uint64_t firstTpHandle = list[0].tpHandle;
    const RdmaHandle rdmaHandle = RdmaHandleManager::GetInstance().GetByIp(devPhyId, param.locAddr);
    if (!rdmaHandle) {
        THROW<InternalException>("[TpManager][%s] can not find rdmaHandle for RaGetTpAttrAsync, devPhyId[%u].",
            __func__, devPhyId);
    }
    void *raReqHandle = nullptr;
    const s32 ret =
        RaGetTpAttrAsync(rdmaHandle, firstTpHandle, &reqCtx.tpAttrBitmap, &reqCtx.tpAttr, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        THROW<NetworkApiException>(StringFormat("[TpManager][StartGetTpAttrForFirstTpDevice] RaGetTpAttrAsync failed "
                                                  "ret[%d] raReqHandle[%p] tpHandle[%llu].",
            ret, raReqHandle, firstTpHandle));
    }
    reqCtx.handle = reinterpret_cast<RequestHandle>(raReqHandle);
    reqCtx.phase = RequestCtx::ReqPhase::WAIT_TP_ATTR;
}

inline TpInfo ParseTpInfo(const struct HccpTpInfo *infoPtr)
{
    TpInfo tpInfo;
    tpInfo.tpHandle = infoPtr->tpHandle;
    return tpInfo;
}

HcclResult TpManager::MapTpInfoFromTpAttr(const RaUbGetTpInfoParam &param, const RequestCtx &reqCtx, TpInfo &outTpInfo) const
{
    const uint32_t tpInfoNum = reqCtx.tpInfoNum;
    const struct HccpTpInfo *baseInfoPtr =
        reinterpret_cast<const struct HccpTpInfo *>(reqCtx.dataBuffer.data());
    const TpQosPolicyInput policy = TpQosPolicyInputFrom(param);
    const uint16_t slMask = TpQosReadSlAvailableMask16(reqCtx.tpAttr);
    const uint32_t slAvailableCnt = TpQosCalSlAvailableCnt(slMask);
    HCCL_INFO("[TpManager][%s] after get_tp_attr: slMask[0x%04x] slAvailableCnt[%u] slBitmap[0x%x] dscp[%u] "
              "dscpConfigMode[%u] tpAttrBitmap[0x%x] param[%s].",
        __func__, static_cast<unsigned>(slMask), slAvailableCnt,
        static_cast<unsigned>(reqCtx.tpAttr.slBitmap), static_cast<unsigned>(reqCtx.tpAttr.dscp & 0x3FU),
        static_cast<unsigned>(reqCtx.tpAttr.dscpConfigMode & 1U), reqCtx.tpAttrBitmap, param.Describe().c_str());

    uint32_t tpListIndex = 0;
    uint32_t mappedSl = 0;
    CHK_RET(TpQosSelectTpListEntry(policy, tpInfoNum, slMask, tpListIndex, mappedSl, "TpManager"));

    outTpInfo = ParseTpInfo(baseInfoPtr + tpListIndex);
    outTpInfo.mappedJettyPriority = mappedSl & 0xFU;
    outTpInfo.hasMappedJettyPriority = true;

    const RdmaHandle ctxHandle = RdmaHandleManager::GetInstance().GetByIp(devPhyId, param.locAddr);
    CHK_PTR_NULL(ctxHandle);
    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboard(devLogicId, isPcieStd));
    CHK_RET(TpQosCommitAttrsAfterSlMapping(ctxHandle, isPcieStd, policy, reqCtx.tpAttr, outTpInfo.tpHandle, mappedSl,
        tpInfoNum, slMask, devPhyId, "TpManager"));

    HCCL_INFO("[TpManager][%s] tp qos mapping ok: tpInfoNum[%u] tpHandle[%llu] tpListIndex[%u] "
              "mappedJettyPriority[%u] qos[%u] param[%s].",
        __func__, tpInfoNum, outTpInfo.tpHandle, tpListIndex, outTpInfo.mappedJettyPriority,
        param.qos & 0xFFU, param.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::HandleCompletedRequest(const TpManager::RequestCtx reqCtx, const RaUbGetTpInfoParam &param,
    TpInfo &tpInfo, bool withSlPolicy)
{
    const uint32_t tpInfoNum = reqCtx.tpInfoNum;
    if (tpInfoNum == 0) {
        HCCL_WARNING("[TpManager][%s] failed to find tp info, tpInfoNum is 0, "
            "param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    HCCL_INFO("[TpManager][%s] RaGetTpInfoList completed: tpInfoNum[%u] withSlPolicy[%d] devPhyId[%u] param[%s].",
        __func__, tpInfoNum, static_cast<int>(withSlPolicy), devPhyId, param.Describe().c_str());

    TpInfo tmpTpInfo{};
    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboard(devLogicId, isPcieStd));
    if (isPcieStd) {
        const struct HccpTpInfo *baseInfoPtr =
            reinterpret_cast<const struct HccpTpInfo *>(reqCtx.dataBuffer.data());
        tmpTpInfo = ParseTpInfo(baseInfoPtr);
        tmpTpInfo.mappedJettyPriority = kPcieStdMappedSl;
        tmpTpInfo.hasMappedJettyPriority = true;
        HCCL_INFO("[TpManager][%s] pcie std mainboard: skip GetTpAttr/SetTpAttr, devPhyId[%u] tpInfoNum[%u] "
                  "mappedSl[%u] tpHandle[%llu] param[%s].",
            __func__, devPhyId, tpInfoNum, kPcieStdMappedSl, tmpTpInfo.tpHandle, param.Describe().c_str());
    } else if (withSlPolicy) {
        CHK_RET(MapTpInfoFromTpAttr(param, reqCtx, tmpTpInfo));
    } else {
        const struct HccpTpInfo *baseInfoPtr =
            reinterpret_cast<const struct HccpTpInfo *>(reqCtx.dataBuffer.data());
        tmpTpInfo = ParseTpInfo(baseInfoPtr);
        tmpTpInfo.hasMappedJettyPriority = false;
    }

    const QosKey qosKey = TpQosMapKeyFromQos(param.qos);

    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto &infoMap = GetInfoCtxMap(param.tpProtocol);
    auto &qosMap = infoMap[param.locAddr][param.rmtAddr];
    auto qIt = qosMap.find(qosKey);
    if (qIt != qosMap.end() && qIt->second.tpInfo.tpHandle == tmpTpInfo.tpHandle) {
        qIt->second.useCnt += 1;
        tpInfo = qIt->second.tpInfo;
    } else {
        qosMap[qosKey] = TpInfoCtx{tmpTpInfo, 1};
        tpInfo = qosMap[qosKey].tpInfo;
    }
    return HcclResult::HCCL_SUCCESS;
}

TpManager::InfoCtxMap& TpManager::GetInfoCtxMap(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpInfoMap;
        case TpProtocol::TP:
            return tpInfoMap;
        case TpProtocol::UBOE:
            return uboeInfoMap;
        default:
            return tpInfoMap;
    }
}

TpManager::ReqCtxMap& TpManager::GetReqCtxMap(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpReqMap;
        case TpProtocol::TP:
            return tpReqMap;
        case TpProtocol::UBOE:
            return uboeReqMap;
        default:
            return tpReqMap;
    }
}

std::mutex& TpManager::GetInfoCtxMutex(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpInfoMutex;
        case TpProtocol::TP:
            return tpInfoMutex;
        case TpProtocol::UBOE:
            return uboeInfoMutex;
        default:
            return tpInfoMutex;
    }
}

std::mutex& TpManager::GetReqCtxMutex(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpReqMutex;
        case TpProtocol::TP:
            return tpReqMutex;
        case TpProtocol::UBOE:
            return uboeReqMutex;
        default:
            return tpReqMutex;
    }
}

} // namespace Hccl
