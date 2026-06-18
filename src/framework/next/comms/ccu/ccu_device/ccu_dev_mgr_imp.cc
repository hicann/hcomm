/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_dev_mgr_imp.h"

#include "hccl_res.h"
#include "hccl_common.h"
#include "eid_info_mgr.h"

// 支持ccu新老通信域混跑临时添加
#include "unified_platform/ccu/ccu_device/ccu_component/ccu_component.h"
#include "unified_platform/ccu/ccu_device/ccu_res_specs.h"
#include "unified_platform/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "exception_handler.h"

/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * #include "ccu_comp.h"
 * #include "ccu_res_specs.h"
 * #include "ccu_res_batch_allocator.h"
 */

// 引入主板类型查询接口，后续应根据ccu驱动提供的信息用于判断
// 当前先简化修改
#include "./ccu_res_specs.h"

#include "adapter_rts.h"

#include "orion_adpt_utils.h"

namespace hcomm {

static std::unordered_map<int32_t, std::shared_ptr<CcuDrvHandle>> ccuDrvHandleMap;
static std::mutex ccuDrvHandleMutex;
static bool ccuDriverInitAgainFlag = false; // 记录每个进程CCU驱动是否重复拉起
static thread_local HcclMainboardId mainBoardType = HcclMainboardId::MAINBOARD_OTHERS; // 记录本卡的主板类型

HcclResult CcuInitFeature(const int32_t devLogicId, std::shared_ptr<CcuDrvHandle> &ccuDrvHandle)
{
    if (devLogicId >= static_cast<int32_t>(MAX_MODULE_DEVICE_NUM)) {
        HCCL_ERROR("[%s] failed, devLogicId[%d] is too large, should be less than %u.",
            __func__, devLogicId, MAX_MODULE_DEVICE_NUM);
        return HcclResult::HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(ccuDrvHandleMutex);
    // ccu驱动已重复拉起失败时，直接返回，在锁保护内返回
    if (ccuDriverInitAgainFlag) {
        return HcclResult::HCCL_E_AGAIN;
    }

    auto iter = ccuDrvHandleMap.find(devLogicId);
    if (iter != ccuDrvHandleMap.end()) {
        ccuDrvHandle = iter->second;
        HCCL_RUN_INFO("[%s] devLogicId[%d] init ccu feature, handle[0x%llx].",
            __func__, devLogicId, ccuDrvHandle.get());
        return HcclResult::HCCL_SUCCESS;
    }

    std::shared_ptr<CcuDrvHandle> drvHandle = nullptr;
    drvHandle.reset(new (std::nothrow) CcuDrvHandle(devLogicId));
    CHK_PTR_NULL(drvHandle);

    auto ret = drvHandle->Init();
    if (ret == HcclResult::HCCL_E_AGAIN) {
        HCCL_WARNING("[%s] failed but passed, ccu driver already be inited, devLogicId[%d].",
            __func__, devLogicId);
        ccuDriverInitAgainFlag = true; // 记录该进程ccu驱动已拉起失败
        drvHandle = nullptr; // 主动置空触发资源销毁，控制释放时序
        return ret;
    }
    CHK_RET(ret);

    ccuDrvHandleMap[devLogicId] = drvHandle;
    ccuDrvHandle = ccuDrvHandleMap[devLogicId];
    HCCL_RUN_INFO("[%s] devLogicId[%d] init ccu feature, handle[0x%llx].",
        __func__, devLogicId, ccuDrvHandle.get());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeinitFeature(const int32_t devLogicId)
{
    std::lock_guard<std::mutex> lock(ccuDrvHandleMutex);
    auto iter = ccuDrvHandleMap.find(devLogicId);
    if (iter == ccuDrvHandleMap.end()) {
        HCCL_INFO("[%s] passed, ccu feature was not be inited, devLogicId[%d].",
            __func__, devLogicId);
        return HcclResult::HCCL_SUCCESS;
    }

    auto &ccuDrvHandle = ccuDrvHandleMap[devLogicId];
    if (ccuDrvHandle.use_count() == 1) {
        HCCL_RUN_INFO("[%s] entry, start to deinit ccu feature, "
            "handle[0x%llx] devLogicId[%d].",
            __func__, ccuDrvHandle.get(), devLogicId);
        ccuDrvHandle = nullptr;
        ccuDrvHandleMap.erase(devLogicId);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuGetDieEnableInfo(int32_t deviceLogicId, uint8_t dieId, bool &enableFlag)
{
    const auto &dieEnableFlags = Hccl::CcuComponent::GetInstance(deviceLogicId).GetDieEnableFlags();
    CHK_PRT_RET(dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR("[%s] failed, dieId[%u] is invalid, shoudle be in [0-%u), devLogicId[%d].",
            __func__, dieId, CCU_MAX_IODIE_NUM, deviceLogicId),
        HcclResult::HCCL_E_PARA);
    enableFlag = dieEnableFlags[dieId];
    return HcclResult::HCCL_SUCCESS;
}

constexpr u32 CCU_MS_DEFAULT_BLOCK_LOOP_ENGINE_REQ = 8 * 8 * 2;
constexpr u32 CCU_MS_DEFAULT_BLOCK_MS_REQ = 64 * 8 * 2;
constexpr u32 CCU_MS_DEFAULT_CKE_REQ = 32;
constexpr u32 CCU_MS_DEFAULT_BLOCK_CKE_REQ = 8 * 8 * 2;
constexpr u32 CCU_MS_DEFAULT_XN_REQ = 400;
constexpr u32 CCU_MS_DEFAULT_GSA_REQ = 400;
constexpr u32 CCU_MS_DEFAULT_MISSIONREQ_REQ = 2;
inline void ConfigCcuResReqCcuMs(CcuResReq &resReq, uint8_t dieId)
{
    resReq.loopEngineReq[dieId] = 0;
    resReq.blockLoopEngineReq[dieId] = CCU_MS_DEFAULT_BLOCK_LOOP_ENGINE_REQ;
    resReq.msReq[dieId] = 0;
    resReq.blockMsReq[dieId] = CCU_MS_DEFAULT_BLOCK_MS_REQ;
    resReq.ckeReq[dieId] = CCU_MS_DEFAULT_CKE_REQ;
    resReq.blockCkeReq[dieId] = CCU_MS_DEFAULT_BLOCK_CKE_REQ;
    resReq.continuousXnReq[dieId] = 0;
    resReq.xnReq[dieId] = CCU_MS_DEFAULT_XN_REQ;
    resReq.gsaReq[dieId] = CCU_MS_DEFAULT_GSA_REQ;
    resReq.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    resReq.missionReq.req[dieId] = CCU_MS_DEFAULT_MISSIONREQ_REQ;
}

constexpr u32 CCU_SCHED_DEFAULT_BLOCK_LOOP_ENGINE_REQ = 16;
constexpr u32 CCU_SCHED_DEFAULT_BLOCK_MS_REQ = 128;
constexpr u32 CCU_SCHED_DEFAULT_CKE_REQ = 32;
constexpr u32 CCU_SCHED_DEFAULT_BLOCK_CKE_REQ = 16;
constexpr u32 CCU_SCHED_DEFAULT_XN_REQ = 400;
constexpr u32 CCU_SCHED_DEFAULT_GSA_REQ = 400;
constexpr u32 CCU_SCHED_DEFAULT_MISSIONREQ_REQ = 2;
inline void ConfigCcuResReqCcuSched(CcuResReq &resReq, uint8_t dieId)
{
    resReq.loopEngineReq[dieId] = 0;
    resReq.blockLoopEngineReq[dieId] = CCU_SCHED_DEFAULT_BLOCK_LOOP_ENGINE_REQ;
    resReq.msReq[dieId] = 0;
    resReq.blockMsReq[dieId] = CCU_SCHED_DEFAULT_BLOCK_MS_REQ;
    resReq.ckeReq[dieId] = CCU_SCHED_DEFAULT_CKE_REQ;
    resReq.blockCkeReq[dieId] = CCU_SCHED_DEFAULT_BLOCK_CKE_REQ;
    resReq.continuousXnReq[dieId] = 0;
    resReq.xnReq[dieId] = CCU_SCHED_DEFAULT_XN_REQ;
    resReq.gsaReq[dieId] = CCU_SCHED_DEFAULT_GSA_REQ;
    resReq.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    resReq.missionReq.req[dieId] = CCU_SCHED_DEFAULT_MISSIONREQ_REQ;
}

// CCU设备管理对集合通信提供的接口
HcclResult CcuAllocEngineResHandle(const int32_t deviceLogicId,
    const CcuEngine ccuEngine, CcuResHandle &resHandle)
{
    const auto &dieEnableFlags = Hccl::CcuComponent::GetInstance(deviceLogicId).GetDieEnableFlags();
    if (!dieEnableFlags[0] && !dieEnableFlags[1]) {
        HCCL_ERROR("[%s] failed, all ccu dies are disable, devLogicId[%d].",
            __func__, deviceLogicId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    CcuResReq resReq{};
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags[dieId]) {
            continue;
        }

        if (ccuEngine == CcuEngine::CCU_MS) {
            ConfigCcuResReqCcuMs(resReq, dieId);
        } else {
            ConfigCcuResReqCcuSched(resReq, dieId);
        }
    }

    if (mainBoardType == HcclMainboardId::MAINBOARD_OTHERS) {
        CHK_RET(CcuGetMainboardId(deviceLogicId, mainBoardType));
    }

    if (mainBoardType == HcclMainboardId::MAINBOARD_PCIE_STD &&
        ccuEngine == CcuEngine::CCU_MS) { // 标卡环境下配置CCU_MS拦截报错
        HCCL_ERROR("[%s] ccuEngine[%s] not support in %s", __func__,
            ccuEngine.Describe().c_str(), mainBoardType.Describe().c_str());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    auto ret = CcuDevMgrImp::AllocResHandle(deviceLogicId, resReq, resHandle);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING("[%s] failed but passed, resource is not enough, "
            "devLogicId[%d], ccuType[%s].", __func__, deviceLogicId,
            ccuEngine.Describe().c_str());
        return ret;
    }
    CHK_RET(ret);

    HCCL_INFO("[%s] succeed, get res handle[%llx], devLogicId[%d]", __func__, resHandle, deviceLogicId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuCheckResource(const int32_t deviceLogicId, const CcuResHandle resHandle, CcuResRepository &resRepo)
{
    CHK_RET(CcuDevMgrImp::GetResource(deviceLogicId, resHandle, resRepo));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuReleaseResHandle(const int32_t deviceLogicId, const CcuResHandle resHandle)
{
    CHK_RET(CcuDevMgrImp::ReleaseResHandle(deviceLogicId, resHandle));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuAllocChannels(const int32_t deviceLogicId, const CcuChannelPara &ccuChannelPara,
    std::vector<CcuChannelInfo> &ccuChannelInfos)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(ccuChannelPara.commAddr, ipAddr)); // 为了打印信息暂时添加
    HCCL_INFO("[%s] new allocation request: deviceLogicId[%d], ipAddr[%s], "
        "channelnum[%u], jettyNum[%u], sqSize[%u].", __func__, deviceLogicId,
        ipAddr.Describe().c_str(), ccuChannelPara.channelNum,
        ccuChannelPara.jettyNum, ccuChannelPara.sqSize);

    uint32_t devPhyId{0};
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devPhyId));

    DevEidInfo eidInfo{};
    CHK_RET(EidInfoMgr::GetInstance(devPhyId).GetEidInfoByAddr(ccuChannelPara.commAddr, eidInfo));
    const uint8_t dieId = static_cast<uint8_t>(eidInfo.dieId);
    const uint32_t feId = eidInfo.funcId;
    ChannelPara para{};
    para.feId = feId;
    para.jettyNum = ccuChannelPara.jettyNum;
    para.sqSize = ccuChannelPara.sqSize;

    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).AllocChannels(dieId, para, ccuChannelInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuReleaseChannel(const int32_t deviceLogicId, const uint8_t dieId,
    const uint32_t ccuChannelId)
{
    HCCL_INFO("[%s] new release request: deviceLogicId[%d], dieId[%u], "
        "ccuChannelId[%u].", __func__, deviceLogicId, dieId, ccuChannelId);

    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ReleaseChannel(dieId, ccuChannelId);
    EXCEPTION_HANDLE_END
    return ret;
}

// 以下为hcomm基础通信内部CCU流程使用的接口
HcclResult CcuDevMgrImp::GetCcuVersion(const int32_t deviceLogicId, CcuVersion &ccuVersion)
{
    EXCEPTION_HANDLE_BEGIN
    ccuVersion = Hccl::CcuResSpecifications::GetInstance(deviceLogicId).GetCcuVersion();
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDevMgrImp::GetCcuResourceSpaceBufInfo(const int32_t deviceLogicId, const uint8_t dieId,
    uint64_t &addr, uint64_t &size)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).GetCcuResourceSpaceBufInfo(dieId, addr, size);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetCcuResourceSpaceTokenInfo(const int32_t deviceLogicId, const uint8_t dieId,
    uint64_t &tokenId, uint64_t &tokenValue)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).GetCcuResourceSpaceTokenInfo(dieId, tokenId, tokenValue);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ConfigChannel(const int32_t deviceLogicId, const uint8_t dieId,
    ChannelCfg &cfg)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ConfigChannel(dieId, cfg);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetLoopChannelId(const int32_t deviceLogicId, const uint8_t srcDieId,
    const uint8_t dstDieId, uint32_t &channIdx)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).GetLoopChannelId(srcDieId, dstDieId, channIdx);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetResource(const int32_t deviceLogicId,
    const CcuResHandle handle, CcuResRepository &ccuResRepo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResBatchAllocator::GetInstance(deviceLogicId).GetResource(handle, ccuResRepo);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocResHandle(const int32_t deviceLogicId, const CcuResReq resReq,
    CcuResHandle &handle)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResBatchAllocator::GetInstance(deviceLogicId).AllocResHandle(resReq, handle);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseResHandle(const int32_t deviceLogicId, const CcuResHandle handle)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResBatchAllocator::GetInstance(deviceLogicId).ReleaseResHandle(handle);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocIns(const int32_t deviceLogicId, const uint8_t dieId,
    const uint32_t num, ResInfo &insInfo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).AllocIns(dieId, num, insInfo);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseIns(const int32_t deviceLogicId, const uint8_t dieId,
    const ResInfo &insInfo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ReleaseIns(dieId, insInfo);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocCke(const int32_t deviceLogicId, const uint8_t dieId,
    const uint32_t num, std::vector<ResInfo> &ckeInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).AllocCke(dieId, num, ckeInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseCke(const int32_t deviceLogicId, const uint8_t dieId,
    const std::vector<ResInfo> &ckeInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ReleaseCke(dieId, ckeInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocXn(const int32_t deviceLogicId, const uint8_t dieId,
    const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).AllocXn(dieId, num, xnInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseXn(const int32_t deviceLogicId, const uint8_t dieId,
    const std::vector<ResInfo> &xnInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ReleaseXn(dieId, xnInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetMissionKey(const int32_t deviceLogicId, const uint8_t dieId,
    uint32_t &missionKey)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResSpecifications::GetInstance(deviceLogicId).GetMissionKey(dieId, missionKey);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetInstructionNum(const int32_t deviceLogicId, const uint8_t dieId,
    uint32_t &instrNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResSpecifications::GetInstance(deviceLogicId).GetInstructionNum(dieId, instrNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetXnBaseAddr(const uint32_t devLogicId, const uint8_t dieId,
    uint64_t& xnBaseAddr)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResSpecifications::GetInstance(devLogicId).GetXnBaseAddr(dieId, xnBaseAddr);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CheckDieValid(const char *funcName, const int32_t devLogicId, const uint8_t dieId,
    const std::array<bool, CCU_MAX_IODIE_NUM> &dieEnableFlags)
{
    CHK_PRT_RET(dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR("[%s] failed, dieId[%u] is invalid, shoudle be in [0-%u), devLogicId[%d].",
            funcName, dieId, CCU_MAX_IODIE_NUM, devLogicId),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(!dieEnableFlags[dieId],
        HCCL_ERROR("[%s] failed, dieId[%u] is disable, devLogicId[%d].",
            funcName, dieId, devLogicId),
        HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuSetTaskKill(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuSetTaskKill] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuSetTaskKill]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).SetTaskKill();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuSetTaskKillDone(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuSetTaskKillDone] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuSetTaskKillDone]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).SetTaskKillDone();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuCleanTaskKillState(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuCleanTaskKillState] Input params: deviceLogicId[%u]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuCleanTaskKillState]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).CleanTaskKillState();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuCleanDieCkes(const int32_t deviceLogicId, const uint8_t dieId)
{
    HCCL_INFO("[CcuCleanDieCkes] Input params: deviceLogicId[%u], dieId[%u]", deviceLogicId, dieId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuCleanDieCkes]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).CleanDieCkes(dieId);
    EXCEPTION_HANDLE_END
    return ret;
}

}; // namespace hcomm