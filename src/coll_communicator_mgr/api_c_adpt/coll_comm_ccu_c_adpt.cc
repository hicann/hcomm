/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_ccu_res.h"

#include <mutex>

#include "hccl_comm_pub.h"

#include "exception_handler.h"
#include "ccu_instance_mgr.h"

/**
 * @note 职责：集合通信的通信域CCU管理的C接口的C到C++适配
 */
HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle* insHandles, uint32_t* insNum)
{
    EXCEPTION_HANDLE_BEGIN

    HcclUs startut = TIME_NOW();

    CHK_PTR_NULL(comm);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const auto& commId = hcclComm->GetIdentifier();
    HCCL_INFO("[%s] CommId[%s] query ccu instance.", __func__, commId.c_str());

    CHK_PTR_NULL(insHandles);
    CHK_PTR_NULL(insNum);

    // CCU不支持A5之前代际
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_WARNING("[%s] is not supported.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    auto* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    auto* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    // 非CCU通信域允许查询，不认为是错误
    auto ccuInsHandle = myRank->GetCcuInstance();
    if (ccuInsHandle == 0) {
        auto opExpansionMode = myRank->GetOpExpansionMode();
        HCCL_WARNING(
            "[%s] failed to get ccu instance, commId[%s] op expansion mode[%u].", __func__, commId.c_str(),
            opExpansionMode);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    insHandles[0] = ccuInsHandle;
    *insNum = 1;
    HCCL_INFO("[%s] success, take time [%lld]us.", __func__, DURATION_US(TIME_NOW() - startut).count());

    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommAssignCcuIns(HcclComm comm, CcuInsHandle insHandle)
{
    EXCEPTION_HANDLE_BEGIN

    HcclUs startut = TIME_NOW();

    CHK_PTR_NULL(comm);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const auto& commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "[%s] CommId[%s] assign ccu instance[%llu].", __func__, commId.c_str(),
        static_cast<unsigned long long>(insHandle));

    if (insHandle == 0) {
        HCCL_ERROR(
            "[%s] failed, commId[%s] insHandle[%llu] is invalid.", __func__, commId.c_str(),
            static_cast<unsigned long long>(insHandle));
        return HcclResult::HCCL_E_PARA;
    }

    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_WARNING("[%s] is not supported.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    auto* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    auto* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    {
        // 仅保证多个 Assign 调用之间并发安全，Query 和通信域销毁由调用方保证不与 Assign 并发。
        static std::mutex assignCcuInsMutex;
        std::lock_guard<std::mutex> lock(assignCcuInsMutex);

        auto oldInsHandle = myRank->GetCcuInstance();
        if (oldInsHandle != 0) {
            HCCL_ERROR(
                "[%s] failed, commId[%s] already has ccu instance[%llu], "
                "new instance[%llu] will not be assigned.",
                __func__, commId.c_str(), static_cast<unsigned long long>(oldInsHandle),
                static_cast<unsigned long long>(insHandle));
            return HcclResult::HCCL_E_PARA;
        }

        const auto devLogicId = collComm->GetDeviceLogicId();
        auto* ccuIns = hcomm::CcuInstanceMgr::GetInstance(devLogicId).Get(insHandle);
        if (ccuIns == nullptr) {
            HCCL_ERROR(
                "[%s] failed, commId[%s] ccu instance[%llu] is not found.", __func__, commId.c_str(),
                static_cast<unsigned long long>(insHandle));
            return HcclResult::HCCL_E_NOT_FOUND;
        }

        myRank->SetCcuInstance(insHandle);
    }

    HCCL_INFO(
        "[%s] success, commId[%s] ccu instance[%llu], take time [%lld]us.", __func__, commId.c_str(),
        static_cast<unsigned long long>(insHandle), DURATION_US(TIME_NOW() - startut));

    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}
