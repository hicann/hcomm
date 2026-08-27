/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_drv_handle.h"

#include "ccu_log.h"

#include "hccp_tlv.h"
#include "hccp_tlv_hdc_mgr.h"

#include "ccu_res_specs.h"
#include "ccu_pfe_cfg_mgr.h"
#include "ccu_comp.h"
#include "ccu_res_batch_allocator.h"
#include "ccu_kernel_mgr.h"

// 支持ccu新老通信域混跑临时添加
#include "unified_platform/ccu/ccu_device/ccu_res_specs_legacy.h"
#include "unified_platform/ccu/ccu_device/ccu_component/ccu_component.h"
#include "unified_platform/ccu/ccu_device/ccu_res_batch_allocator_legacy.h"
#include "unified_platform/ccu/ccu_context/ccu_context_mgr_imp.h"
#include "hccp_tlv_hdc_manager.h"

#include "exception_handler.h"

namespace hcomm {

inline bool CheckCcuOpenSourceEnable()
{
    // A6 不支持legacy ccu mc2，可以完全切换至开源流程
    auto devType = DevType::DEV_TYPE_COUNT;
    (void)hrtGetDeviceType(devType);
    return devType == DevType::DEV_TYPE_960;
}

static HcclResult HccpRaTlvRequest(const TlvHandle tlvHandle, const u32 tlvModuleType, const u32 tlvCcuMsgType)
{
    CHK_PTR_NULL(tlvHandle);
    struct TlvMsg sendMsg {};
    struct TlvMsg recvMsg {};
    sendMsg.type = tlvCcuMsgType;

    HCCL_INFO("[%s] tlvHandle[%p].", __func__, tlvHandle);
    constexpr u32 RA_TLV_REQUEST_UNAVAIL = 128308;
    int32_t ret = RaTlvRequest(tlvHandle, tlvModuleType, &sendMsg, &recvMsg);
    if (ret == RA_TLV_REQUEST_UNAVAIL || ret == OTHERS_ENOTSUPP) {
        HCCL_RUN_WARNING(
            "[%s] ra tlv request UNAVAIL, tlvHandle[%p], tlvModeulType[%u], tlvCcuMsgType[%u], ret[%d].", __func__,
            tlvHandle, tlvModuleType, tlvCcuMsgType, ret);
        return HCCL_E_AGAIN; // 代表CCU驱动已被拉起，需要等待其他进程退出
    }

    if (ret != 0) {
        HCCL_ERROR(
            "[Request][RaTlv]errNo[0x%016llx] ra tlv request fail. "
            "return: ret[%d], module type[%u], message type[%u]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), tlvModuleType, tlvCcuMsgType);
        return HcclResult::HCCL_E_NETWORK;
    }

    HCCL_INFO(
        "tlv request success, tlv module type[%u], "
        "message type[%u]",
        tlvModuleType, tlvCcuMsgType);
    return HcclResult::HCCL_SUCCESS;
}

CcuResult CcuDrvHandle::Init()
{
    HCCL_RUN_INFO("[CcuDrvHandle][%s], deviceLogicId: %d", __func__, devLogicId_);
    CCU_CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(devLogicId_), devPhyId_));
    // 支持ccu新老通信域混跑
    CCU_EXCEPTION_HANDLE_BEGIN
    // 初始化CCU平台层能力，有时序要求
    // 当前走进A5通信域，暂时不需要主动拉起HDC通道
    /* 为了支持ccu新老通信域混跑，暂时复用原有的tlv mgr，避免重复申请资源
     * auto &tlvHdcMgr = HccpTlvHdcMgr::GetInstance(devPhyId_);
     * CHK_RET(tlvHdcMgr.Init());
     * tlvHandle_ = tlvHdcMgr.GetHandle();
     */

    tlvHandle_ = Hccl::HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId_);
    CCU_CHK_PTR_NULL(tlvHandle_);
    // 拉起CCU驱动如果因其他进程占用重复拉起时，返回EAGAIN，日志检查返回值打印warning
    auto ret = HccpRaTlvRequest(tlvHandle_, TLV_MODULE_TYPE_CCU, MSG_TYPE_CCU_INIT);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        HCCL_RUN_WARNING(
            "[%s] HccpRaTlvRequest ret[%d], repeat init ccu, deviceLogicId[%d].", __func__, ret, devLogicId_);
        return CcuResult::CCU_E_DRV_BUSY;
    }
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[%s] failed to init ccu driver, ret[%d] is unexpected.", __func__, ret);
        return CcuResult::CCU_E_DRV_INIT_FAILED;
    }

    if (!CheckCcuOpenSourceEnable()) {
        Hccl::CcuResSpecifications::GetInstance(devLogicId_).Init();
        Hccl::CcuComponent::GetInstance(devLogicId_).Init();
        Hccl::CcuResBatchAllocator::GetInstance(devLogicId_).Init();
        Hccl::CtxMgrImp::GetInstance(devLogicId_).Init();
    } else {
        CCU_CHK_RET(CcuResSpecifications::GetInstance(devLogicId_).Init());
        CCU_CHK_RET(CcuPfeCfgMgr::GetInstance(devLogicId_).Init());
        CCU_CHK_RET(CcuComponent::GetInstance(devLogicId_).Init());
        CCU_CHK_RET(CcuResBatchAllocator::GetInstance(devLogicId_).Init());
    }

    CCU_CHK_RET(CcuKernelMgr::GetInstance(devLogicId_).Init());

    CCU_EXCEPTION_HANDLE_END

    return CcuResult::CCU_SUCCESS;
}

static HcclResult CcuLegacyMgrDeinit(int32_t devLogicId)
{
    // 释放有时序要求
    EXCEPTION_HANDLE_BEGIN
    Hccl::CtxMgrImp::GetInstance(devLogicId).Deinit();
    Hccl::CcuResBatchAllocator::GetInstance(devLogicId).Deinit();
    Hccl::CcuComponent::GetInstance(devLogicId).Deinit();
    Hccl::CcuResSpecifications::GetInstance(devLogicId).Deinit();
    EXCEPTION_HANDLE_END

    return HcclResult::HCCL_SUCCESS;
}

CcuResult CcuDrvHandle::Deinit()
{
    // 释放流程不打断，不抛异常，尽量尝试释放所有资源
    // 释放有时序要求
    HCCL_RUN_INFO("[CcuDrvHandle] start to deinit ccu driver, deviceLogicId[%d].", devLogicId_);
    (void)CcuKernelMgr::GetInstance(devLogicId_).Deinit();

    if (!CheckCcuOpenSourceEnable()) {
        (void)CcuLegacyMgrDeinit(devLogicId_);
    } else {
        (void)CcuResBatchAllocator::GetInstance(devLogicId_).Deinit();
        (void)CcuComponent::GetInstance(devLogicId_).Deinit();
        (void)CcuPfeCfgMgr::GetInstance(devLogicId_).Deinit();
        (void)CcuResSpecifications::GetInstance(devLogicId_).Deinit();
    }

    if (tlvHandle_ != nullptr) {
        (void)HccpRaTlvRequest(tlvHandle_, TLV_MODULE_TYPE_CCU, MSG_TYPE_CCU_UNINIT);
        tlvHandle_ = nullptr;
    }

    return CcuResult::CCU_SUCCESS;
}

CcuDrvHandle::~CcuDrvHandle() { (void)Deinit(); }

} // namespace hcomm
