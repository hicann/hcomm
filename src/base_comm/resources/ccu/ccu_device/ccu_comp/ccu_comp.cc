/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_comp.h"

#include <random>

#include "hccl_common.h"
#include "rdma_handle_manager.h"

#include "eid_info_mgr.h"
#include "ccu_res_specs.h"
#include "ccu_channel_ctx_mgr_v1.h"
#include "ccu_channel_ctx_mgr_v2.h"

#include "exception_handler.h"
#include "adapter_rts_common.h"
#include "env_config.h"
#include "orion_adapter_hccp.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

constexpr TpProtocol LOOP_JETTY_PROTOCOL = TpProtocol::RTP; // 环回使用RTP避免被环境link down阻塞
constexpr uint8_t CCU_MAX_MISSION_NUM = 16;

// 设置为0，分配数量由channelCtxMgr决定，v1 默认1个
constexpr uint32_t LOOP_CHANNEL_USE_JETTY  = 0;
constexpr uint32_t LOOP_CHANNEL_USE_SQSIZE_V1 = 16;
constexpr uint32_t LOOP_CHANNEL_USE_SQSIZE_V2 = 32;

// 环回获取TP信息超时等待10s
constexpr uint32_t LOOP_CHANNEL_WAIT_TIMEOUT_MS = 10000;

// 环境是ARM+X86时，配置 die0 的 MS 交织粒度为 1<<7 = 128
constexpr uint32_t MSID_CONFIG_ARMX86_MAINBOARD = 7;
// 设计支持的最大IOdie数量
constexpr uint8_t MAX_CCU_IODIE_NUM = 2;
// 清理CKE批量申请大小
constexpr u32 MAX_CKE_DATA_ARRAY_SIZE = 8;

CcuComponent &CcuComponent::GetInstance(const int32_t deviceLogicId)
{
    static CcuComponent ccuComponent[MAX_MODULE_DEVICE_NUM + 1];
    int32_t devLogicId = deviceLogicId;
    if (devLogicId < 0 || static_cast<uint32_t>(devLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[CcuComponent][%s] use the backup device, devLogicId[%d] should be "
            "less than %u.", __func__, devLogicId, MAX_MODULE_DEVICE_NUM);
        devLogicId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    ccuComponent[devLogicId].devLogicId_ = devLogicId;
    return ccuComponent[devLogicId];
}

HcclResult CcuComponent::Init()
{
    std::lock_guard<std::mutex> _lock(innerMutex_);

    if (initFlag_) {
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(devLogicId_), devPhyId_));
    CHK_RET(CheckDiesEnable());
    CHK_RET(CreateCcuRmaBuffer());
    CHK_RET(CreateResourceManagers());
    CHK_RET(CreateLoopChannels());
    CHK_RET(ConfigMsIdToken());
    initFlag_ = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::Deinit()
{
    std::lock_guard<std::mutex> _lock(innerMutex_);
    CHK_RET(ReleaseJettyRes());

    loopFeCommAddrMap_.clear();
    ccuRmaBufferMap_.clear();
    
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        channelCtxMgrs_[dieId] = nullptr;
        resAllocators_[dieId] = nullptr;
        loopChannelIds_[dieId] = INVAILD_LOOP_CHANNEL_ID;
    }

    initFlag_ = false;
    return HcclResult::HCCL_SUCCESS;
}

CcuComponent::~CcuComponent()
{
    (void)Deinit();
}

static std::array<bool, CCU_MAX_IODIE_NUM> GetDieDrvEnableFlags(const int32_t devLogicId)
{
    // 根据资源规格的记录驱动可用的die
    std::array<bool, CCU_MAX_IODIE_NUM> dieDrvEnableFlags{false, false}; 
    const auto &ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId);
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        (void)ccuResSpecs.GetDieEnableFlag(dieId, dieDrvEnableFlags[dieId]);
        if (!dieDrvEnableFlags[dieId]) { // 调用接口失败时不会改变dieEnableFlags[i]
            HCCL_WARNING("[CcuComponent][%s] devLogicId[%d], dieId[%u] driver is not usable.",
                __func__, devLogicId, dieId);
        }
    }

    return dieDrvEnableFlags;
}

HcclResult CcuComponent::CheckDiesEnable()
{
    ccuVersion_ = CcuResSpecifications::GetInstance(devLogicId_).GetCcuVersion();
    HCCL_INFO("[CcuComponent][%s] ccu version[%s], devLogicId[%d].",
        __func__, ccuVersion_.Describe().c_str(), devLogicId_);

    const auto &dieDrvEnableFlags = GetDieDrvEnableFlags(devLogicId_);
    // 内部检查驱动可用的die上是否配置eid，内部更新die是否可用的标记
    CHK_RET(ChooseLoopEids(dieDrvEnableFlags));

    bool allDieDisable = true;
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        allDieDisable = allDieDisable && !dieEnableFlags_[dieId];
    }

    if (allDieDisable) {
        HCCL_ERROR("[CcuComponent][%s] failed, because all dies are "
            "disabled, devLogicId[%d].", __func__, devLogicId_);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    return HcclResult::HCCL_SUCCESS;
}

static HcclResult FindOneUsableEid(const int32_t devLogicId, const uint32_t devPhyId,
    const uint8_t dieId, uint32_t &feId, CommAddr &commAddr)
{
    // 如果无法查询设备是否为uboe设备，报错退出
    CHK_RET(HccpGetUboeFlagEnable(devPhyId));

    std::vector<DevEidInfo> eidInfos;
    auto ret = EidInfoMgr::GetInstance(devPhyId).GetEidInfos(eidInfos);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, devLogicId[%d], dieId[%u].",
            __func__, devLogicId, dieId),
        ret);

    std::string name;
    bool findFlag = false;
    // 当前结论，除仅包含UBOE的FE外
    // 其他eid均支持源与目标eid一致时应用环回
    // 故当前版本选择首个可用eid即可
    EXCEPTION_HANDLE_BEGIN
    auto &rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    for (auto &eidInfo : eidInfos) {
        // 如果是UBOE设备或非本die，则跳过
        if (HccpCheckUboeSupported(eidInfo.devFeature) || (eidInfo.dieId != dieId)) {
            continue;
        }

        Hccl::IpAddress ipAddr{};
        CHK_RET(CommAddrToIpAddress(eidInfo.commAddr, ipAddr));
        const auto rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, ipAddr);
        CHK_PTR_NULL(rdmaHandle);
        const bool rtpEnable = rdmaHandleMgr.GetRtpEnable(rdmaHandle);
        if (!rtpEnable) {
            // 遍历端口可能较多，避免刷屏不打印
            continue;
        }

        feId = eidInfo.funcId;
        commAddr = eidInfo.commAddr;
        name = eidInfo.name;
        findFlag = true;
    }
    EXCEPTION_HANDLE_END

    if (!findFlag) {
        HCCL_WARNING("[CcuComponent][%s] dieId[%u] doesn't have usable func ID, "
            "devLogicId[%d].", __func__, dieId, devLogicId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr, ipAddr));
    HCCL_INFO("[CcuComponent][%s] dieId[%u] choose: name[%s] feId[%u] ipAddr[%s], "
        "devLogicId[%d].", __func__, dieId, name.c_str(), feId,
        ipAddr.Describe().c_str(), devLogicId);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ChooseLoopEids(const std::array<bool, CCU_MAX_IODIE_NUM> &dieDrvEnableFlags)
{
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieDrvEnableFlags[dieId]) {
            dieEnableFlags_[dieId] = false;
            continue;
        }

        uint32_t feId = 0;
        CommAddr commAddr{};
        if (FindOneUsableEid(devLogicId_, devPhyId_, dieId, feId, commAddr) != HcclResult::HCCL_SUCCESS) {
            dieEnableFlags_[dieId] = false;
            HCCL_WARNING("[CcuComponent][%s] failed to find feId eid, but passed, "
                "devLogicId[%d], dieId[%u].", __func__, devLogicId_, dieId);
            continue;
        }

        loopFeCommAddrMap_[dieId] = {feId, commAddr};
        dieEnableFlags_[dieId] = true;
        HCCL_RUN_INFO("[CcuComponent][%s] devLogicId[%d] die[%u] is usable.",
            __func__, devLogicId_, dieId);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetLoopFeIpByDieId(const uint8_t dieId, uint32_t &feId,
    CommAddr &commAddr)
{
    const auto &dieIter = loopFeCommAddrMap_.find(dieId);
    CHK_PRT_RET(dieIter == loopFeCommAddrMap_.end(),
        HCCL_WARNING("[CcuComponent][%s] failed but passed, "
            "dieId[%u] doesn't have usable loop feId, devLogicId[%d].",
            __func__, dieId, devLogicId_),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto &feIdCommAddr = dieIter->second;
    feId = feIdCommAddr.first;
    commAddr = feIdCommAddr.second;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::CreateCcuRmaBuffer()
{
    auto &rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    auto &ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId_);
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags_[dieId]) {
            continue;
        }

        uint32_t feId = 0;
        CommAddr commAddr{};
        if (GetLoopFeIpByDieId(dieId, feId, commAddr) != HcclResult::HCCL_SUCCESS) {
            continue;
        }

        uint64_t ccuResAddr = 0;
        (void)ccuResSpecs.GetResourceAddr(dieId, ccuResAddr);
        if (ccuResAddr == 0) {
            HCCL_WARNING("[CcuComponent][%s] failed, ccu resource space address[0] is invalid, "
                "devLogicId[%d] dieId[%u]", __func__, devLogicId_, dieId);
            continue;
        }

        // 申请rdmaHandle可能抛异常
        EXCEPTION_HANDLE_BEGIN
        Hccl::IpAddress ipAddr{};
        CHK_RET(CommAddrToIpAddress(commAddr, ipAddr));
        const CtxHandle ctxHandle = static_cast<CtxHandle>(rdmaHandleMgr.GetByIp(devPhyId_, ipAddr));
        CHK_PTR_NULL(ctxHandle);
        const auto ccuBuffer = std::make_shared<Hccl::Buffer>(ccuResAddr, CCU_RESOURCE_SIZE);
        ccuRmaBufferMap_.emplace(dieId,
            std::make_unique<Hccl::LocalUbRmaBuffer>(ccuBuffer, ctxHandle));

        EXCEPTION_HANDLE_END
    }

    return HcclResult::HCCL_SUCCESS;
}

static HcclResult CreateChannelCtxMgrByVersion(const CcuVersion version,
    const uint32_t devLogicId, const uint8_t dieId, const uint32_t devPhyId,
    std::unique_ptr<CcuChannelCtxMgr>& channelCtxMgr)
{
    switch (version) {
        case CcuVersion::CCU_V1:
            channelCtxMgr.reset(
                new (std::nothrow) CcuChannelCtxMgrV1(devLogicId, dieId, devPhyId));
            break;
         case CcuVersion::CCU_V2:
            channelCtxMgr.reset(
                new (std::nothrow) CcuChannelCtxMgrV2(devLogicId, dieId, devPhyId));
            break;
        default:
            HCCL_ERROR("[CcuComponent][%s] failed, ccu driver version[%s] is not expected, "
                "devLogicId[%d] dieId[%u].", __func__, version.Describe().c_str(),
                devLogicId, dieId);
            return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(channelCtxMgr);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::CreateResourceManagers()
{
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags_[dieId]) {
            continue;
        }

        std::unique_ptr<CcuChannelCtxMgr> channelCtxMgrPtr = nullptr;
        CHK_RET(CreateChannelCtxMgrByVersion(ccuVersion_, devLogicId_,
            dieId, devPhyId_, channelCtxMgrPtr));
        CHK_RET(channelCtxMgrPtr->Init());

        std::unique_ptr<CcuResAllocator> resAllocatorPtr = nullptr;
        resAllocatorPtr.reset(new (std::nothrow) CcuResAllocator(devLogicId_, dieId));
        CHK_PTR_NULL(resAllocatorPtr);
        CHK_RET(resAllocatorPtr->Init());

        channelCtxMgrs_[dieId] = std::move(channelCtxMgrPtr);
        resAllocators_[dieId] = std::move(resAllocatorPtr);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::CreateLoopChannels()
{
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        loopChannelIds_[dieId] = INVAILD_LOOP_CHANNEL_ID;
        // 失败抛异常处理，jetty资源跟随数据结构析构释放
        auto ret = CreateLoopChannel(dieId, loopChannelIds_[dieId]);
        CHK_PRT_RET(ret,
           HCCL_ERROR("[CcuComponent][%s] failed, devLogicId[%d], dieId[%u].",
            __func__, devLogicId_, dieId),
            ret);

        if (loopChannelIds_[dieId] == INVAILD_LOOP_CHANNEL_ID) {
            HCCL_RUN_WARNING("[CcuComponent][%s] failed but passed, loop channel id[%u], "
                "devLogicId[%d], dieId[%u].", __func__, loopChannelIds_[dieId],
                devLogicId_, dieId);
            continue;
        }

        HCCL_RUN_INFO("[CcuComponent][%s] succeed, loop channel id[%u], "
            "devLogicId[%d], dieId[%u].", __func__, loopChannelIds_[dieId],
            devLogicId_, dieId);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::CreateLoopChannel(const uint8_t dieId, uint32_t &channelId)
{
    if (!dieEnableFlags_[dieId]) {
        HCCL_WARNING("CcuComponent][%s] passed, dieId[%u] is not enable, "
            "devLogicId[%d].", __func__, dieId, devLogicId_);
        return HcclResult::HCCL_SUCCESS;
    }

    // 对于单p或单die场景，可能设备或die不会配置eid，按成功处理不阻塞用例
    uint32_t feId = 0;
    CommAddr commAddr{};
    if (GetLoopFeIpByDieId(dieId, feId, commAddr) != HcclResult::HCCL_SUCCESS) {
        channelId = INVAILD_LOOP_CHANNEL_ID;
        HCCL_WARNING("[CcuComponent][%s] failed but passed, dieId[%u] doesn't have loop feId, "
            "devLogicId[%d].", __func__, dieId, devLogicId_);
        return HcclResult::HCCL_SUCCESS;
    }
    const uint32_t loopChannelSqsize = (ccuVersion_ == CcuVersion::CCU_V1 ?
        LOOP_CHANNEL_USE_SQSIZE_V1 : LOOP_CHANNEL_USE_SQSIZE_V2);
    std::vector<ChannelInfo> channelInfos; // 按jetty组分配
    const ChannelPara channelPara{feId, LOOP_CHANNEL_USE_JETTY, loopChannelSqsize};
    auto ret = channelCtxMgrs_[dieId]->Alloc(channelPara, channelInfos);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed to alloc channel, "
            "devLogicId[%d], dieId[%u].", __func__, devLogicId_, dieId),
        ret);

    const auto &channelInfo = channelInfos[0]; // 环回只使用1个channel
    ret = CreateAndImportLoopJettys(dieId, commAddr, channelInfo.jettyInfos);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed to create or import loop jettys, "
            "devLogicId[%d], dieId[%u].", __func__, devLogicId_, dieId),
        ret);

    ret = ConfigLoopChannel(dieId, commAddr, channelInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed to config the loop channel, "
            "devLogicId[%d], dieId[%u].", __func__, devLogicId_, dieId),
        ret);

    channelId = channelInfo.channelId;
    return HcclResult::HCCL_SUCCESS;
}

JettyImportCfg GetJettyImportCfg(const TpInfo &tpInfo, const uint32_t &psn)
{
    const TpHandle tpHandle = tpInfo.tpHandle;
    HCCL_INFO("[CcuComponent][%s] loop channel use tp handle[%llu] psn[%u].",
        __func__, tpHandle, psn);

    JettyImportCfg cfg = {};
    cfg.localTpHandle = tpHandle;
    cfg.remoteTpHandle = tpHandle;
    cfg.localPsn = psn;
    cfg.remotePsn = psn;
    cfg.protocol = LOOP_JETTY_PROTOCOL;
    return cfg;
}

HcclResult CcuComponent::CreateAndImportLoopJettys(const uint8_t dieId,
    const CommAddr &commAddr, const std::vector<JettyInfo> &jettyInfos)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr, ipAddr));

    Hccl::CqCreateInfo cqInfo{0};
    auto &rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    const auto ctxHandle = static_cast<CtxHandle>(rdmaHandleMgr.GetByIp(devPhyId_, ipAddr));
    const auto _jfcHandle = rdmaHandleMgr.GetJfcHandle(ctxHandle, cqInfo, Hccl::HrtUbJfcMode::CCU_POLL);
    const JfcHandle jfcHandle = reinterpret_cast<JfcHandle>(_jfcHandle);

    const auto &rmaBufferIter = ccuRmaBufferMap_.find(dieId);
    CHK_PRT_RET(rmaBufferIter == ccuRmaBufferMap_.end(),
        HCCL_RUN_WARNING("[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].", __func__, dieId, devLogicId_),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto &ccuRmaBuffer = rmaBufferIter->second;
    const auto ccuBufTokenValue = ccuRmaBuffer->GetTokenValue();
    const auto tokenIdHandle = reinterpret_cast<void *>(ccuRmaBuffer->GetTokenIdHandle());
    
    auto &createdVec = createdOutParamMap_[dieId];
    auto &importedVec = importedOutParamMap_[dieId];

    TpInfo loopTpInfo{};
    CHK_RET(GetLoopTpInfo(dieId, commAddr, loopTpInfo));
    const uint32_t loopJettyQos = loopTpInfo.hasMappedJettyPriority
        ? (loopTpInfo.mappedJettyPriority & 0xFU)
        : EnvConfig::UB_QOS_DEFAULT;

    TpAttrInfo tpAttrInfo{};
    CHK_RET(GetLoopTpAttr(dieId, commAddr, tpAttrInfo));
    const uint8_t errTimeout = TpMgr::CalcTaTimeout(tpAttrInfo);

    for (const auto &jettyInfo : jettyInfos) {
        const auto jettyMode = jettyInfo.jettyType == CcuJettyType::CCUM_CACHED_JETTY ?
            HrtJettyMode::CCU_CCUM_CACHE : HrtJettyMode::CCU_TA_CACHE;
        HrtRaUbCreateJettyParam req{jfcHandle, jfcHandle, ccuBufTokenValue,
            tokenIdHandle, jettyMode, jettyInfo.taJettyId, jettyInfo.sqBufVa,
            jettyInfo.sqBufSize, jettyInfo.wqeBBStartId, jettyInfo.sqDepth,
            errTimeout};
        req.qos = loopJettyQos;

        HrtRaUbJettyCreatedOutParam createdOutParam{};
        CHK_RET(HccpUbCreateJetty(ctxHandle, req, createdOutParam));
        createdVec.emplace_back(createdOutParam);

        const auto psn = GetNewPsn();
        const auto &jettyImportCfg = GetJettyImportCfg(loopTpInfo, psn);

        HrtRaUbJettyImportedOutParam importedOutParam{};
        CHK_RET(HccpUbTpImportJetty(ctxHandle, createdOutParam.key,
            createdOutParam.keySize, ccuBufTokenValue, jettyImportCfg, importedOutParam));
        importedVec.emplace_back(std::make_pair(ctxHandle, importedOutParam));
    }

    return HcclResult::HCCL_SUCCESS;
}

static GetTpInfoParam MakeLoopGetTpInfoParam(const CommAddr &commAddr)
{
    GetTpInfoParam param;
    param.locAddr = commAddr;
    param.rmtAddr = commAddr;
    param.tpProtocol = LOOP_JETTY_PROTOCOL;
    param.qos = 0U; // CCU 环回与通信域 hcclQos 解耦；SL 仅由 RaGetTpAttr.slBitmap + loopFirstTpLowestSl 决定
    param.slLevelCount = 0;
    param.loopFirstTpLowestSl = true;
    param.ccuLoopbackGetTpInfo = true;
    return param;
}

static HcclResult RequestNewLoopTpInfo(const uint32_t devPhyId,
    const CommAddr &commAddr, TpInfo &tpInfo)
{
    constexpr auto timeout = std::chrono::milliseconds(LOOP_CHANNEL_WAIT_TIMEOUT_MS);
    const auto startTime = std::chrono::steady_clock::now();

    auto &tpMgr = TpMgr::GetInstance(devPhyId);
    const GetTpInfoParam &tpParam = MakeLoopGetTpInfoParam(commAddr);
    HcclResult ret = HcclResult::HCCL_SUCCESS;
    do {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[CcuComponent][%s] failed, get tp info "
                "timeout[%d ms], devPhyId[%u].", __func__, timeout, devPhyId);
            return HcclResult::HCCL_E_TIMEOUT;
        }

        ret = tpMgr.GetTpInfo(tpParam, tpInfo);
    } while (ret == HcclResult::HCCL_E_AGAIN);

    CHK_RET(ret); // 非重试属于异常情况
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetLoopTpInfo(const uint8_t dieId,
    const CommAddr &commAddr, TpInfo &tpInfo)
{
    const auto &srcIter = tpInfoMap_.find(dieId);
    // 优先使用已经创建过的tpHandle
    if (srcIter == tpInfoMap_.end()) {
        TpInfo newTpInfo{};
        CHK_RET(RequestNewLoopTpInfo(devPhyId_, commAddr, newTpInfo));
        tpInfoMap_[dieId] = std::move(newTpInfo);
    }

    tpInfo = tpInfoMap_[dieId];
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult RequestNewLoopTpAttr(const uint32_t devPhyId, CtxHandle ctxHandle,
    const TpHandle tpHandle, TpAttrInfo &tpAttrInfo)
{
    constexpr auto timeout = std::chrono::milliseconds(LOOP_CHANNEL_WAIT_TIMEOUT_MS);
    const auto startTime = std::chrono::steady_clock::now();

    auto &tpMgr = TpMgr::GetInstance(devPhyId);
    constexpr uint32_t TP_ATTR_BITMAP = 0;
    const GetTpAttrParam tpAttrParam = {tpHandle, TP_ATTR_BITMAP};
    HcclResult ret = HcclResult::HCCL_SUCCESS;
    do {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[CcuComponent][%s] failed, get tp attr "
                "timeout[%d ms], devPhyId[%d].", __func__, timeout, devPhyId);
            return HcclResult::HCCL_E_TIMEOUT;
        }

        ret = tpMgr.GetTpAttr(tpAttrParam, tpAttrInfo, ctxHandle);
    } while (ret == HcclResult::HCCL_E_AGAIN);

    CHK_RET(ret);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetLoopTpAttr(const uint8_t dieId,
    const CommAddr &commAddr, TpAttrInfo &tpAttrInfo)
{
    const auto &srcIter = tpAttrInfoMap_.find(dieId);
    if (srcIter == tpAttrInfoMap_.end()) {
        const auto &tpInfoIter = tpInfoMap_.find(dieId);
        CHK_PRT_RET(tpInfoIter == tpInfoMap_.end(),
            HCCL_ERROR("[CcuComponent][%s] failed, tpInfo not found for dieId[%u], "
                "devLogicId[%d].", __func__, dieId, devLogicId_),
            HcclResult::HCCL_E_NOT_FOUND);

        Hccl::IpAddress ipAddr{};
        CHK_RET(CommAddrToIpAddress(commAddr, ipAddr));
        auto &rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
        const CtxHandle ctxHandle = static_cast<CtxHandle>(rdmaHandleMgr.GetByIp(devPhyId_, ipAddr));

        TpAttrInfo newTpAttrInfo{};
        CHK_RET(RequestNewLoopTpAttr(devPhyId_, ctxHandle, tpInfoIter->second.tpHandle, newTpAttrInfo));
        tpAttrInfoMap_[dieId] = std::move(newTpAttrInfo);
    }

    tpAttrInfo = tpAttrInfoMap_[dieId];
    return HcclResult::HCCL_SUCCESS;
}

inline uint32_t GenerateRandomNum()
{
    uint32_t randNum = std::rand();
    return randNum;
}

uint32_t CcuComponent::GetNewPsn()
{
    return GenerateRandomNum();
}

HcclResult CcuComponent::ConfigLoopChannel(const uint8_t dieId, const CommAddr &commAddr,
    const ChannelInfo &channelInfo)
{
    const uint32_t dstDieId = 1 - dieId; // 当前仅存在最多两个die
    // 当前环回复用支持die内die间，当两个die均启用时应配置对die，否则为本die
    auto rmaBufferIter = ccuRmaBufferMap_.find(dstDieId);
    if (rmaBufferIter == ccuRmaBufferMap_.end()) {
        rmaBufferIter = ccuRmaBufferMap_.find(dieId);
    }

    CHK_PRT_RET(rmaBufferIter == ccuRmaBufferMap_.end(),
        HCCL_WARNING("[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].", __func__, dieId, devLogicId_),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto &ccuRmaBuffer = rmaBufferIter->second;
    const auto ccuBufTokenValue = ccuRmaBuffer->GetTokenValue();

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr, ipAddr));

    ChannelCfg cfg{};
    cfg.channelId = channelInfo.channelId;
    CHK_RET(IpAddressToReverseHcclEid(ipAddr, cfg.remoteEid));
    cfg.tpn       = importedOutParamMap_[dieId][0].second.tpn; // 环回仅1个对端
    cfg.remoteCcuVa   = ccuRmaBuffer->GetBuf()->GetAddr();
    cfg.memTokenId    = ccuRmaBuffer->GetTokenId();
    cfg.memTokenValue = ccuBufTokenValue;

    const auto &jettyInfos = channelInfo.jettyInfos;
    const auto &createdVec = createdOutParamMap_[dieId];
    const uint32_t jettyNum = jettyInfos.size();
    for (uint32_t i = 0; i < jettyNum; i++) {
        cfg.jettyCfgs.emplace_back(JettyCfg{
            jettyInfos[i].jettyCtxId,
            createdVec[i].dbVa,
            createdVec[i].dbTokenId,
            ccuBufTokenValue
        });
    }

    return channelCtxMgrs_[dieId]->Config(cfg);
}

HcclResult CcuComponent::ConfigMsIdToken()
{
    const auto serveMode = CcuResSpecifications::GetInstance(devLogicId_).GetServeMode();
    CustomChannelInfoIn  inBuff{};
    CustomChannelInfoOut outBuff{};
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        const auto &dieIter = ccuRmaBufferMap_.find(dieId);
        if (dieIter == ccuRmaBufferMap_.end()) {
            HCCL_WARNING("[CcuComponent][%s] failed but passed, ccu rma buffer of die[%u] "
                "is not existed, devLogicId[%d].", __func__, dieId, devLogicId_);
            continue;
        }
        const auto &ccuRmaBuffer = dieIter->second;
        const uint32_t tokenId = ccuRmaBuffer->GetTokenId();
        const uint32_t tokenValue = ccuRmaBuffer->GetTokenValue();
        uint32_t msId = 0;
        // 非A+X, 非die 0，采用默认交织粒度
        if (serveMode == ServeMode::ARMX86 && dieId == 0) {
            msId = MSID_CONFIG_ARMX86_MAINBOARD;
        } else {
            CHK_RET(CcuResSpecifications::GetInstance(devLogicId_).GetMsId(dieId, msId));
        }

        inBuff.op                    = CcuOpcodeType::CCU_U_OP_SET_MSID_TOKEN;
        inBuff.offsetStartIdx        = 0;
        inBuff.data.dataInfo.udieIdx = dieId;
        inBuff.data.dataInfo.dataArray[0].baseinfo.msId       = msId;
        inBuff.data.dataInfo.dataArray[0].baseinfo.tokenId    = tokenId;
        inBuff.data.dataInfo.dataArray[0].baseinfo.tokenValue = tokenValue;

        auto ret = HccpRaTlvCcuCustomChannel(devLogicId_,
        static_cast<void *>(&inBuff), static_cast<void *>(&outBuff));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CcuComponent][%s] failed to call ccu driver, "
                "devLogicId[%d] dieId[%d] op[%s] ret[%d].", __func__, devLogicId_, dieId,
                "SET_MSID_TOKEN", ret);
            return ret;
        }

        HCCL_INFO("[CcuComponent][%s] config MS ID token success, dieId[%u], msid[%u]",
            __func__, dieId, msId);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetCcuResourceSpaceBufInfo(const uint8_t dieId, uint64_t &addr,
    uint64_t &size) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    auto res = ccuRmaBufferMap_.find(dieId);
    CHK_PRT_RET(res == ccuRmaBufferMap_.end(),
        HCCL_WARNING("[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].", __func__, dieId, devLogicId_),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto rawBuffer = res->second->GetBuf();
    addr = static_cast<uint64_t>(rawBuffer->GetAddr());
    size = static_cast<uint64_t>(rawBuffer->GetSize());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetCcuResourceSpaceTokenInfo(const uint8_t dieId, uint64_t &tokenId,
    uint64_t &tokenValue) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    auto res = ccuRmaBufferMap_.find(dieId);
    CHK_PRT_RET(res == ccuRmaBufferMap_.end(),
        HCCL_WARNING("[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].", __func__, dieId, devLogicId_),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto &ccuRmaBuffer = res->second;
    tokenId = static_cast<uint64_t>(ccuRmaBuffer->GetTokenId());
    tokenValue = static_cast<uint64_t>(ccuRmaBuffer->GetTokenValue());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocChannels(const uint8_t dieId, const ChannelPara &channelPara,
    std::vector<ChannelInfo> &channelInfos)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(channelCtxMgrs_[dieId]);
    auto ret = channelCtxMgrs_[dieId]->Alloc(channelPara, channelInfos);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, feId[%u], devLogicId[%d], dieId[%u].",
            __func__, channelPara.feId, devLogicId_, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ConfigChannel(const uint8_t dieId, const ChannelCfg &cfg)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    uint32_t channelId = cfg.channelId;
    CHK_PRT_RET(channelId == loopChannelIds_[dieId],
        HCCL_WARNING("[CcuComponent][%s] failed, refused to config loop channel[%u], "
            "devLogicId[%d], dieId[%u].", __func__, channelId, devLogicId_, dieId),
        HcclResult::HCCL_E_PARA);

    CHK_PTR_NULL(channelCtxMgrs_[dieId]);
    auto ret = channelCtxMgrs_[dieId]->Config(cfg);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, channelId[%u], devLogicId[%d], dieId[%u].",
            __func__, channelId, devLogicId_, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseChannel(const uint8_t dieId, const uint32_t channelId)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));
    CHK_PRT_RET(channelId == loopChannelIds_[dieId],
        HCCL_WARNING("[CcuComponent][%s] failed, refused to release loop channel[%u], "
            "devLogicId[%d], dieId[%u].", __func__, channelId, devLogicId_, dieId),
        HcclResult::HCCL_E_PARA);

    CHK_PTR_NULL(channelCtxMgrs_[dieId]);
    auto ret = channelCtxMgrs_[dieId]->Release(channelId);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, channelId[%u], devLogicId[%d], dieId[%u].",
            __func__, channelId, devLogicId_, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetLoopChannelId(const uint8_t srcDieId, const uint8_t dstDieId,
    uint32_t &channelId) const
{
    channelId = INVAILD_LOOP_CHANNEL_ID; // 允许die未启用时查询环回channelId

    CHK_RET(CheckDieValid(__func__, devLogicId_, srcDieId, {true, true}));
    CHK_RET(CheckDieValid(__func__, devLogicId_, dstDieId, {true, true}));

    // 特殊处理die未启用场景
    CHK_PRT_RET(!dieEnableFlags_[srcDieId] || !dieEnableFlags_[dstDieId],
        HCCL_WARNING("[CcuComponent][%s] passed, srcDie[%u] or dstDie[%u] is not enable,"
            "devLogicId[%d].", __func__, srcDieId, dstDieId, devLogicId_),
        HcclResult::HCCL_SUCCESS);

    // 当前环回channel每个die占用1个，不区分die内die间
    CHK_PRT_RET(loopChannelIds_[srcDieId] == INVAILD_LOOP_CHANNEL_ID,
        HCCL_ERROR("[CcuComponent][%s] failed, invalid loop channel id, "
            "devLogicId[%d], srcDieId[%u].", __func__, devLogicId_, srcDieId),
        HcclResult::HCCL_E_INTERNAL);

    channelId = loopChannelIds_[srcDieId];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocRes(const uint8_t dieId, const ResType resType, const uint32_t num,
    const bool consecutive, std::vector<ResInfo> &resInfos)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(resAllocators_[dieId]);
    auto ret = resAllocators_[dieId]->Alloc(resType, num, consecutive, resInfos);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, resType[%s], num[%u], devLogicId[%d], dieId[%u].",
            __func__, resType.Describe().c_str(), num, devLogicId_, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseRes(const uint8_t dieId, const ResType resType, const uint32_t startId,
    const uint32_t num)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(resAllocators_[dieId]);
    auto ret = resAllocators_[dieId]->Release(resType, startId, num);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, resType[%s], startId[%u], num[%u], "
            "devLogicId[%d], dieId[%u].", __func__, resType.Describe().c_str(),
            startId, num, devLogicId_, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocIns(const uint8_t dieId, const uint32_t num, ResInfo &insInfo)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(resAllocators_[dieId]);
    std::vector<ResInfo> resInfos;
    auto ret = resAllocators_[dieId]->Alloc(ResType::INS, num, true, resInfos);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, num[%u], devLogicId[%d], dieId[%u].",
            __func__, num, devLogicId_, dieId),
        ret);

    insInfo = resInfos[0]; // 申请连续资源只会有一份
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseIns(const uint8_t dieId, const ResInfo &insInfo)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(resAllocators_[dieId]);
    auto ret = resAllocators_[dieId]->Release(ResType::INS, insInfo.startId, insInfo.num);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, resInfo[%s], devLogicId[%d], dieId[%u].",
            __func__, insInfo.Describe().c_str(), devLogicId_, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocCke(const uint8_t dieId, const uint32_t num, std::vector<ResInfo> &ckeInfos)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(resAllocators_[dieId]);
    auto ret = resAllocators_[dieId]->Alloc(ResType::CKE, num, false, ckeInfos);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, num[%u], devLogicId[%d], dieId[%u].",
            __func__, num, devLogicId_, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseCke(const uint8_t dieId, const std::vector<ResInfo> &ckeInfos)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(resAllocators_[dieId]);
    for (auto &ckeInfo : ckeInfos) {
        auto ret = resAllocators_[dieId]->Release(ResType::CKE, ckeInfo.startId, ckeInfo.num);
        CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
            HCCL_WARNING("[CcuComponent][%s] failed, resInfo[%s], devLogicId[%d], dieId[%u].",
                __func__, ckeInfo.Describe().c_str(), devLogicId_, dieId),
            ret);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocXn(const uint8_t dieId, const uint32_t num, std::vector<ResInfo> &xnInfos)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(resAllocators_[dieId]);
    auto ret = resAllocators_[dieId]->Alloc(ResType::XN, num, false, xnInfos);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, num[%u], devLogicId[%d], dieId[%u].",
            __func__, num, devLogicId_, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseXn(const uint8_t dieId, const std::vector<ResInfo> &xnInfos)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    CHK_PTR_NULL(resAllocators_[dieId]);
    for (auto &xnInfo : xnInfos) {
        auto ret = resAllocators_[dieId]->Release(ResType::XN, xnInfo.startId, xnInfo.num);
        CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
            HCCL_WARNING("[CcuComponent][%s] failed, resInfo[%s], devLogicId[%d], dieId[%u].",
                __func__, xnInfo.Describe().c_str(), devLogicId_, dieId),
            ret);
    }

    return HcclResult::HCCL_SUCCESS;
}

constexpr u32 WISH_COUNT_XN_NUM = 511;
constexpr u32 TOTAL_COUNT_XN_NUM = 1;

HcclResult CcuComponent::SetSplitUnit(uint8_t dieId, uint32_t splitPktUnit) const
{
    CHK_PRT_RET(dieId >= MAX_CCU_IODIE_NUM,
        HCCL_ERROR("[CcuComponent][%s] failed, dieId[%u] is invalid, shoudle be in [0-%u), devLogicId[%d].",
            __func__, dieId, MAX_CCU_IODIE_NUM, devLogicId_),
        HcclResult::HCCL_E_PARA);

    CustomChannelInfoIn  inBuff{};
    CustomChannelInfoOut outBuff{};

    inBuff.op                          = CcuOpcodeType::CCU_U_OP_SET_TIF_SPLIT_SIZE;
    inBuff.data.dataInfo.udieIdx       = dieId;
    inBuff.data.dataInfo.dataArraySize = 1;
    inBuff.data.dataInfo.dataLen       = sizeof(CcuDataTypeUnion) * inBuff.data.dataInfo.dataArraySize;

    inBuff.data.dataInfo.dataArray[0].tifSplitSize.splitPktUnit = splitPktUnit & 0b1;
    inBuff.data.dataInfo.dataArray[0].tifSplitSize.tpSplitSize  = 0x2;    // 0x2:TP模式的拆包size为4KB
    inBuff.data.dataInfo.dataArray[0].tifSplitSize.ctpSplitSize = 0x1;    // 0x1:CTP模式的拆包size为4KB

    auto ret = HccpRaTlvCcuCustomChannel(devLogicId_,
        static_cast<void *>(&inBuff), static_cast<void *>(&outBuff));
    if (ret != 0) {
        HCCL_ERROR("[CcuComponent][%s] failed to call ccu driver, "
            "devPhyId[%u] dieId[%d] op[%s].", __func__, devPhyId_, dieId,
            "CCU_U_OP_SET_TIF_SPLIT_SIZE");
        return HcclResult::HCCL_E_NETWORK;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetAvailableTotalCntXnIndex(uint32_t& index) const
{
    for (uint32_t i = 0; i < CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM; ++i) {
        if (!usedTotalCntXnFlags_[i]) {
            index = i;
            return HcclResult::HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[CcuComponent][%s] failed, no available TotalCnt Xns.", __func__);
    return HcclResult::HCCL_E_UNAVAIL;
}

HcclResult CcuComponent::SetTotalCntXn(uint8_t dieId, uint32_t fromId, uint32_t toId, uint32_t totalId, uint32_t index)
{
    CHK_PRT_RET(fromId > toId,
        HCCL_ERROR("[CcuComponent][%s] failed, fromId or toId invalid, fromId[%u] > toId[%u].", __func__, fromId, toId),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(fromId <= totalId && totalId <= toId,
        HCCL_ERROR("[CcuComponent][%s] failed, totalId[%u] invalid, should not be in [fromId[%u], toId[%u]].",
            __func__, totalId, fromId, toId),
        HcclResult::HCCL_E_PARA);

    HcclResult ret = SetTotalCntXnProcess(dieId, index, fromId, toId, totalId);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CcuComponent][%s] failed, dieId[%u], index[%u], devLogicId[%d].",
            __func__, dieId, index, devLogicId_),
        ret);

    usedTotalCntXnFlags_[index] = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ResetTotalCntXn(uint8_t dieId, uint32_t index)
{
    if (index >= CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM || !usedTotalCntXnFlags_[index]) {
        return HcclResult::HCCL_SUCCESS;
    }

    static constexpr uint32_t fromIdDefault = 0xFFFF;   // from默认值
    static constexpr uint32_t toIdDefault   = 0x0000;   // to默认值
    static constexpr uint32_t totalIdDefault[CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM] {0x3FFC, 0x3FFD, 0x3FFE, 0x3FFF};   // total默认值

    auto ret = SetTotalCntXnProcess(dieId, index, fromIdDefault, toIdDefault, totalIdDefault[index]);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CcuComponent][%s] failed, dieId[%u], index[%u], devLogicId[%d].",
            __func__, dieId, index, devLogicId_),
        ret);

    usedTotalCntXnFlags_[index] = false;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::SetTotalCntXnProcess(uint8_t dieId, uint32_t index, uint32_t fromId, uint32_t toId, uint32_t totalId) const
{
    CHK_PRT_RET(dieId >= MAX_CCU_IODIE_NUM,
        HCCL_ERROR("[CcuComponent][%s] failed, dieId[%u] is invalid, shoudle be in [0-%u), devLogicId[%d].",
            __func__, dieId, MAX_CCU_IODIE_NUM, devLogicId_),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(index >= CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM,
        HCCL_ERROR("[CcuComponent][%s] failed, index[%u] is invalid, shoudle be in [0-%u), devLogicId[%d].",
            __func__, index, CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM, devLogicId_),
        HcclResult::HCCL_E_PARA);

    CustomChannelInfoIn  inBuff{};
    CustomChannelInfoOut outBuff{};

    inBuff.op                          = CcuOpcodeType::CCU_U_OP_SET_XN_TOTAL_CNT;
    inBuff.data.dataInfo.udieIdx       = dieId;
    inBuff.data.dataInfo.dataArraySize = 1;
    inBuff.data.dataInfo.dataLen       = sizeof(CcuDataTypeUnion) * inBuff.data.dataInfo.dataArraySize;

    inBuff.data.dataInfo.dataArray[0].xnTotalCnt.cntIndex     = index & 0b11;    // range: [0, 3]
    inBuff.data.dataInfo.dataArray[0].xnTotalCnt.flagFromAddr = fromId;
    inBuff.data.dataInfo.dataArray[0].xnTotalCnt.flagToAddr   = toId;
    inBuff.data.dataInfo.dataArray[0].xnTotalCnt.totalAddr    = totalId;
    auto ret = HccpRaTlvCcuCustomChannel(devLogicId_,
        static_cast<void *>(&inBuff), static_cast<void *>(&outBuff));
    if (ret != 0) {
        HCCL_ERROR("[CcuComponent][%s] failed to call ccu driver, "
            "devPhyId[%u] dieId[%d] op[%s].", __func__, devPhyId_, dieId,
            "CCU_U_OP_SET_XN_TOTAL_CNT");
        return HcclResult::HCCL_E_NETWORK;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ConfirmCntXns(const uint8_t dieId, const std::string &resGroupTag,
    const ResInfo &cntXnInfos)
{
    struct CntXnBlock cntXnBlock;
    uint32_t totalCntXnId = cntXnInfos.startId + cntXnInfos.num - TOTAL_COUNT_XN_NUM;
    uint32_t wishCntXnIdBegin = cntXnInfos.startId;
    uint32_t wishCntXnIdEnd = totalCntXnId - 1;
    uint32_t blockIdx = CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM;  // invalid value

    HCCL_INFO("Set TotalCntXn, wishCntXnIdBegin[%u] wishCntXnIdEnd[%u] totalCntXnId[%u]",
        wishCntXnIdBegin, wishCntXnIdEnd, totalCntXnId);

    auto ret = GetAvailableTotalCntXnIndex(blockIdx);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CcuComponent][%s] failed, no available TotalCnt Xns, dieId[%u], devLogicId[%d].",
            __func__, dieId, devLogicId_),
        ret);
    CHK_RET(SetTotalCntXn(dieId, wishCntXnIdBegin, wishCntXnIdEnd, totalCntXnId, blockIdx));
    HCCL_INFO("Set TotalCntXn success, index[%u]", blockIdx);

    ret = SetSplitUnit(dieId, 0);  //0表示stomic store add value的单位是byte。1表示以包为单位
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuComponent][%s] SetSplitUnit failed, dieId[%u], devLogicId[%d].",
        __func__, dieId, devLogicId_);
        CHK_RET(ResetTotalCntXn(dieId, blockIdx));
        return ret;
    }

    for (u32 idx = wishCntXnIdBegin; idx <= wishCntXnIdEnd; idx++) {
        cntXnBlock.wishCntXns.push(idx);
    }
    cntXnBlock.resInfo = cntXnInfos;
    cntXnBlock.totalCntXn = totalCntXnId;
    cntXnBlock.blockIdx = blockIdx;
    cntXnBlocks_[dieId].insert(std::make_pair(resGroupTag, cntXnBlock));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocWishCntXn(const uint8_t dieId, const std::string &resGroupTag,
    uint32_t &wishCntXn)
{
    CHK_PRT_RET((ccuVersion_ != CcuVersion::CCU_V2),
        HCCL_ERROR("[CcuComponent][%s] failed, ccuVersion[%d] does not support this interface.",
            __func__, ccuVersion_), HCCL_E_NOT_SUPPORT);
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    std::unique_lock<std::mutex> lock(cntXnBlockMutex_);
    auto &cntXnBlocks = cntXnBlocks_[dieId];
    auto iter = cntXnBlocks.find(resGroupTag);
    if (iter != cntXnBlocks.end()) {
        CHK_PRT_RET((iter->second.wishCntXns.size() == 0),
            HCCL_ERROR("[CcuComponent][%s] failed, wishCntXn is not enough, resGroupTag[%s], devLogicId[%d], "
                "dieId[%u].", __func__, resGroupTag.c_str(), devLogicId_, dieId), HCCL_E_UNAVAIL);
    } else {
        CHK_PRT_RET((cntXnBlocks.size() == CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM),
            HCCL_ERROR("[CcuComponent][%s] failed, cntXnBlock is not enough, resGroupTag[%s], "
                "devLogicId[%d], dieId[%u].", __func__, resGroupTag.c_str(), devLogicId_, dieId), HCCL_E_UNAVAIL);
        ResInfo countXnInfo;
        // 申请511 + 1个cntXn，前511个为wishCntXn，最后一个为totalCntXn
        auto ret = resAllocators_[dieId]->AllocCountXn(WISH_COUNT_XN_NUM + TOTAL_COUNT_XN_NUM, countXnInfo);
        CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuComponent][%s] failed, num[%u], resGroupTag[%s], devLogicId[%d], dieId[%u].",
                __func__, (WISH_COUNT_XN_NUM + TOTAL_COUNT_XN_NUM), resGroupTag.c_str(), devLogicId_, dieId), ret);
        // 配置cntXn
        ret = ConfirmCntXns(dieId, resGroupTag, countXnInfo);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[CcuComponent][%s] failed[%d] to confirm cnt xns, "
                "try to release new allocated cnt xns, dieId[%u] resGroupTag[%s].",
                __func__, ret, dieId, resGroupTag.c_str());
            CHK_RET(resAllocators_[dieId]->ReleaseCountXn(countXnInfo.startId, countXnInfo.num));
            return ret;
        }
    }
    auto &xnBlock = cntXnBlocks_[dieId][resGroupTag];
    HCCL_INFO("resGroupTag[%s]stack size[%u]", resGroupTag.c_str(), xnBlock.wishCntXns.size());
    wishCntXn = xnBlock.wishCntXns.top();
    xnBlock.wishCntXns.pop();
    uint32_t totalCntXn = xnBlock.totalCntXn;
    HCCL_INFO("[CcuComponent][%s] success, resGroupTag[%s], devLogicId[%d], dieId[%u], wishCntXn[%u], totalCntXn[%u].",
        __func__, resGroupTag.c_str(), devLogicId_, dieId, wishCntXn, totalCntXn);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseWishCntXn(const uint8_t dieId, const std::string &resGroupTag, uint32_t wishCntXn)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    std::unique_lock<std::mutex> lock(cntXnBlockMutex_);
    if (cntXnBlocks_[dieId].find(resGroupTag) == cntXnBlocks_[dieId].end()) {
        HCCL_ERROR("[CcuComponent][%s] failed, resGroupTag[%s] is not found, devLogicId[%d], dieId[%u].",
            __func__, resGroupTag.c_str(), devLogicId_, dieId);
        return HCCL_E_NOT_FOUND;
    }

    auto &xnBlock = cntXnBlocks_[dieId][resGroupTag];
    xnBlock.wishCntXns.push(wishCntXn);
    if (xnBlock.wishCntXns.size() != WISH_COUNT_XN_NUM) {
        HCCL_INFO("[CcuComponent][%s] success, resGroupTag[%s], devLogicId[%d], dieId[%u], wishCntXn[%u], available "
            "wishCntXn num[%u].",
            __func__, resGroupTag.c_str(), devLogicId_, dieId, wishCntXn, xnBlock.wishCntXns.size());
        return HCCL_SUCCESS;
    }

    // 所有wishCnt都已经release，释放资源
    CHK_RET(ResetTotalCntXn(dieId, xnBlock.blockIdx));

    auto ret = resAllocators_[dieId]->ReleaseCountXn(xnBlock.resInfo.startId,
        xnBlock.resInfo.num);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CcuComponent][%s] failed, resGroupTag[%s], resInfo[%s], devLogicId[%d], dieId[%u].",
            __func__, resGroupTag.c_str(), xnBlock.resInfo.Describe().c_str(), devLogicId_, dieId),
        ret);
    cntXnBlocks_[dieId].erase(resGroupTag);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetCntXnBlock(const uint8_t dieId, const std::string &resGroupTag,
    std::pair<uint32_t, uint32_t> &cntXnPair)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    std::unique_lock<std::mutex> lock(cntXnBlockMutex_);
    auto iter = cntXnBlocks_[dieId].find(resGroupTag);
    if (iter == cntXnBlocks_[dieId].end()) {
        HCCL_ERROR("[CcuComponent][%s] failed, resGroupTag[%s] is not found, devLogicId[%d], dieId[%u].",
            __func__, resGroupTag.c_str(), devLogicId_, dieId);
        return HCCL_E_NOT_FOUND;
    }

    cntXnPair = std::make_pair(iter->second.resInfo.startId,
        iter->second.totalCntXn);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetTotalCntXn(const uint8_t dieId,
    const std::string &resGroupTag, uint32_t &totalCntXn)
{
    CHK_RET(CheckDieValid(__func__, devLogicId_, dieId, dieEnableFlags_));

    std::unique_lock<std::mutex> lock(cntXnBlockMutex_);
    auto iter = cntXnBlocks_[dieId].find(resGroupTag);
    if (iter == cntXnBlocks_[dieId].end()) {
        HCCL_ERROR("[CcuComponent][%s] failed, resGroupTag[%s] is not found, devLogicId[%d], dieId[%u].",
            __func__, resGroupTag.c_str(), devLogicId_, dieId);
        return HCCL_E_NOT_FOUND;
    }

    totalCntXn = iter->second.totalCntXn;

    return HcclResult::HCCL_SUCCESS;
}

const std::array<bool, CCU_MAX_IODIE_NUM> &CcuComponent::GetDieEnableFlags() const
{
    return dieEnableFlags_;
}

HcclResult CcuComponent::ReleaseJettyRes()
{
    CHK_RET(UnimportAllJettys());
    CHK_RET(ReleaseAllTpInfos());
    CHK_RET(DestroyAllJettys());
    // HrtRaUbLocalMemReg 跟随 LocalUbRmaBuffer 析构时释放
    // 环回channel不需要手动释放，channelCtxMgr跟随CcuComponent释放
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::UnimportAllJettys()
{
    for (auto &importedVec : importedOutParamMap_) {
        for (auto &paramPair : importedVec.second) {
            const auto ctxHandle = paramPair.first;
            const auto remoteJettyHandle = paramPair.second.handle;
            if (!ctxHandle || !remoteJettyHandle) {
                continue;
            }
            int32_t ret = RaCtxQpUnimport(ctxHandle, remoteJettyHandle);
            if (ret != 0) {
                HCCL_ERROR("[CcuComponent][%s] failed, ctxHandle[%p] "
                    "remoteJettyHandle[%p], devLogicId[%d].", __func__,
                    ctxHandle, remoteJettyHandle, devLogicId_);
            }
            paramPair.second.handle = 0; // 清理handle，避免重复释放
        }
    }
    importedOutParamMap_.clear();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseAllTpInfos()
{
    for (auto &item : tpAttrInfoMap_) {
        const auto &dieId = item.first;
        const auto &tpAttrInfo = item.second;
        const auto &tpInfoIter = tpInfoMap_.find(dieId);
        if (tpInfoIter != tpInfoMap_.end() && tpInfoIter->second.tpHandle != 0) {
            (void)TpMgr::GetInstance(devPhyId_).ReleaseTpAttr(tpInfoIter->second.tpHandle, tpAttrInfo);
        }
    }
    tpAttrInfoMap_.clear();
    for (auto &item : tpInfoMap_) {
        const auto &dieId = item.first;
        const auto &tpInfo = item.second;
        if (!tpInfo.tpHandle) {
            continue;
        }

        const auto &dieIdIter = loopFeCommAddrMap_.find(dieId);
        if (dieIdIter == loopFeCommAddrMap_.end()) {
            HCCL_ERROR("[CcuComponent][%s] failed, dieId[%u] loop comm address"
                " is not found, devLogicId[%d].", __func__,
                static_cast<uint32_t>(dieId), devLogicId_);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        const auto &commAddr = dieIdIter->second.second;
        const GetTpInfoParam &tpParam = MakeLoopGetTpInfoParam(commAddr);
        (void)TpMgr::GetInstance(devPhyId_).ReleaseTpInfo(tpParam, tpInfo);
        item.second.tpHandle = 0; // 清理handle，避免重复释放
    }
    tpInfoMap_.clear();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::DestroyAllJettys()
{
    for (auto &createdVec : createdOutParamMap_) {
        for (auto &param : createdVec.second) {
            const auto jettyHandle = param.handle;
            if (!jettyHandle) {
                continue;
            }
            int32_t ret = RaCtxQpDestroy(jettyHandle);
            if (ret != 0) {
                HCCL_ERROR("[CcuComponent][%s] failed, jettyHandle[%p], "
                    "devLogicId[%d].", __func__, jettyHandle, devLogicId_);
            }
            param.handle = 0; // 清理handle，避免重复释放
        }
    }
    createdOutParamMap_.clear();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::SetProcess(CcuOpcodeType opCode) const
{
    CustomChannelInfoIn  inBuff;
    CustomChannelInfoOut outBuff;

    inBuff.op = opCode;
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        if (!dieEnableFlags_[dieId]) {
            HCCL_WARNING("[%s]devLogicId[%d], dieId[%u] is not enable, skip." , __func__, devLogicId_, dieId);
            continue;
        }
        HCCL_INFO("[%s]devLogicId[%d], dieId[%u] start.", __func__, devLogicId_, dieId);
        inBuff.data.dataInfo.udieIdx = dieId;
        auto ret = HccpRaTlvCcuCustomChannel(devLogicId_,
            static_cast<void *>(&inBuff), static_cast<void *>(&outBuff));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] failed to call ccu driver, devLogicId[%d] dieId[%u] op[%u] ret[%d].",
                __func__, devLogicId_, dieId, static_cast<uint32_t>(opCode), ret),
            ret);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::CleanTaskKillState() const
{
    CHK_RET(SetProcess(CcuOpcodeType::CCU_U_OP_CLEAN_TASKKILL_STATE));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::SetTaskKill()
{
    std::lock_guard<std::mutex> _lock(taskKillMutex_);// 加锁，确保线程安全
    
    // 初始化状态下，设置任务kill状态
    if (status == CcuTaskKillStatus::INVALID) {
        status = CcuTaskKillStatus::INIT;
    }

    if (status == CcuTaskKillStatus::TASK_KILL) {
        HCCL_INFO("No need to set task kill, state = %u, devLogicId = %u", status, devLogicId_);
        return HcclResult::HCCL_SUCCESS;
    }

    if (status != CcuTaskKillStatus::INIT) {
        HCCL_ERROR("[CcuComponent][%s] failed, cannot be invoked in the current state, "
            "state = %u, devLogicId = %d.", __func__, status, devLogicId_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    CHK_RET(SetProcess(CcuOpcodeType::CCU_U_OP_SET_TASKKILL));
    status = CcuTaskKillStatus::TASK_KILL;
    HCCL_INFO("[CcuComponent][%s] success, state = %u, devLogicId = %d.", __func__, status, devLogicId_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::SetTaskKillDone()
{
    std::lock_guard<std::mutex> _lock(taskKillMutex_);// 加锁，确保线程安全
    if (status == CcuTaskKillStatus::INVALID) {
        HCCL_ERROR("[CcuComponent][%s] failed, cannot be invoked in the current state, "
            "state = %u, devLogicId = %d.", __func__, status, devLogicId_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (status == CcuTaskKillStatus::INIT) {
        HCCL_INFO("No need to set task kill done, state = %u, devLogicId = %u", status, devLogicId_);
        return HcclResult::HCCL_SUCCESS;
    }

    if (status != CcuTaskKillStatus::TASK_KILL) {
        HCCL_ERROR("[CcuComponent][%s] failed, cannot be invoked in the current state, "
            "state = %u, devLogicId = %d.", __func__, status, devLogicId_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    CHK_RET(SetProcess(CcuOpcodeType::CCU_U_OP_CLEAN_TASKKILL_STATE));
    status = CcuTaskKillStatus::INIT;
    HCCL_INFO("[CcuComponent][%s] success, state = %u, devLogicId = %d", __func__, status, devLogicId_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::CcuSetTaskKillDone(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuSetTaskKillDone] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuSetTaskKillDone]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    return CcuComponent::GetInstance(deviceLogicId).SetTaskKillDone();
}

HcclResult CcuComponent::CcuCleanTaskKillState(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuCleanTaskKillState] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuCleanTaskKillState]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    return CcuComponent::GetInstance(deviceLogicId).CleanTaskKillState();
}

// 以下接口用于n秒快恢与TaskException
HcclResult CcuComponent::CleanDieCkes(const uint8_t dieId) const
{
    CHK_PRT_RET(dieId >= MAX_CCU_IODIE_NUM,
        HCCL_WARNING("[%s] failed, dieId[%u] is invalid, should be in [0-%u), devLogicId[%d].",
        __func__, dieId, MAX_CCU_IODIE_NUM, devLogicId_), HcclResult::HCCL_E_PARA);

    if (!dieEnableFlags_[dieId]) {
        HCCL_INFO("[%s] dieId[%u] is not enable, skip", __func__, dieId);
        return HcclResult::HCCL_SUCCESS;
    }

    CustomChannelInfoIn  inBuff{};
    CustomChannelInfoOut outBuff{};

    // 设置操作码和数据
    uint32_t ckeNum = 0;
    CHK_RET(CcuResSpecifications::GetInstance(devLogicId_).GetCkeNum(dieId, ckeNum));
    HCCL_INFO("[CcuComponent][CleanAllCke]Nsrecovery devLogicId[%d], dieId[%u] ckeNum[%u].",
        devLogicId_, dieId, ckeNum);
    
    inBuff.op                          = CcuOpcodeType::CCU_U_OP_SET_CKE;
    inBuff.data.dataInfo.udieIdx       = dieId;
    // 接口限制，目前方案每次最多清理8个cke，超过8个时分多次清理
    for (uint32_t startIdx = 0; startIdx < ckeNum; startIdx += MAX_CKE_DATA_ARRAY_SIZE) {
        inBuff.data.dataInfo.dataArraySize = std::min(ckeNum - startIdx, MAX_CKE_DATA_ARRAY_SIZE);
        inBuff.data.dataInfo.dataLen       = sizeof(CcuDataByte8) * inBuff.data.dataInfo.dataArraySize;
        inBuff.offsetStartIdx              = startIdx;
        auto ret = HccpRaTlvCcuCustomChannel(devLogicId_,
            static_cast<void *>(&inBuff), static_cast<void *>(&outBuff));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] failed to call ccu driver, devLogicId[%d] dieId[%u] op[%s] ret[%d].",
                __func__, devLogicId_, dieId, "SET_CKE", ret),
            ret);
    }

    return HcclResult::HCCL_SUCCESS;
}

}; // namespace hcomm