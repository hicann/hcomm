/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_container.h"

#include "log.h"

#include "hcom_common.h"
#include "exception_handler.h"

#include "ccu_dev_mgr.h"
#include "ccu_res_pack.h"
#include "ccu_kernel_mgr.h"

namespace hcomm {

constexpr uint32_t DEFAULT_MODE = 0;
constexpr uint32_t CCU_MS_MODE = 5;
constexpr uint32_t CCU_SCHE_MODE = 6;
inline CcuEngine OpExpansionModeToCcuEngine(const uint32_t opExpansionMode)
{
    if (opExpansionMode == DEFAULT_MODE ||
        opExpansionMode == CCU_SCHE_MODE) {
        return CcuEngine::CCU_SCHE;
    }

    if (opExpansionMode == CCU_MS_MODE) {
        return CcuEngine::CCU_MS;
    }
    
    return CcuEngine::INVALID;
}

CcuResContainer::~CcuResContainer()
{
    // 主动释放资源保证时序，不得随意调整顺序
    for (auto &kernelHandle : kernelHandles_) {
        if (kernelHandle != 0) {
            (void)CcuKernelMgr::GetInstance(devLogicId_).UnRegister(kernelHandle);
            kernelHandle = 0;
        }
    }
    kernelHandles_.clear();

    resPack_ = nullptr; // 释放通信域持有CCU资源
    if (ccuDrvHandle_) {
        ccuDrvHandle_ = nullptr;
        (void)CcuDeinitFeature(devLogicId_);
        // 尝试关闭CCU功能，最后一个通信域调用时会关闭CCU驱动
    }
}

HcclResult CcuResContainer::ChangeMode(uint32_t opExpansionMode)
{
    if (opExpansionMode_ != opExpansionMode) {
        opExpansionMode_ = opExpansionMode;
        auto ret = Init();
        if (ret == HcclResult::HCCL_E_AGAIN || ret == HcclResult::HCCL_E_UNAVAIL) {
            HCCL_RUN_WARNING("[%s] init ret[%d] need fallback, opExpansionMode[%d].", __func__, ret, opExpansionMode_);
            return ret;
        }
        CHK_RET(ret);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResContainer::Init()
{
    const auto ccuEngine = OpExpansionModeToCcuEngine(opExpansionMode_);
    if (ccuEngine == CcuEngine::INVALID) {
        // 主动销毁资源，有时序要求，需要先释放ccu资源后关闭驱动
        resPack_ = nullptr;
        ccuDrvHandle_ = nullptr;
        return HcclResult::HCCL_SUCCESS;
    }

    devLogicId_ = HcclGetThreadDeviceId();

    if (!ccuDrvHandle_) {
        auto ret = CcuInitFeature(devLogicId_, ccuDrvHandle_); // 重复拉起会返回HCCL_E_AGAIN
        if (ret != HcclResult::HCCL_SUCCESS) {
            ccuDrvHandle_ = nullptr;
            return ret;
        }
    }

    if (!resPack_) {
        resPack_.reset(new (std::nothrow) CcuResPack(ccuEngine));
        CHK_PTR_NULL(resPack_);
        auto ret = resPack_->Init(); // 资源不足会返回HCCL_E_UNAVAIL
        if (ret != HcclResult::HCCL_SUCCESS) {
            // 避免重复拉起ccu驱动，此时ccuDrvHandle不置空
            // ccu驱动生命周期仍跟随ccuResContainer
            // 等外部销毁ccuResContainer再释放
            resPack_ = nullptr;
            return ret;
        }
    }
    // 除重复拉起和资源不足这两种需要回退的情况外，不允许其它错误返回HCCL_E_AGAIN和HCCL_E_UNAVAIL错误码

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResContainer::ResetResPack()
{
    if (!resPack_) {
        return HcclResult::HCCL_SUCCESS;
    }

    untranslatedKernelHandles_.clear();
    CHK_RET(resPack_->Reset());
    return HcclResult::HCCL_SUCCESS;
}

CcuResPack *CcuResContainer::GetResPack()
{
    return resPack_.get();
}

HcclResult CcuResContainer::SaveCcuKernel(const CcuKernelHandle kernelHandle)
{
    EXCEPTION_HANDLE_BEGIN
    kernelHandles_.push_back(kernelHandle);
    untranslatedKernelHandles_.push_back(kernelHandle);
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

const std::vector<CcuKernelHandle> &CcuResContainer::GetUntranslatedKernels()
{
    return untranslatedKernelHandles_;
}

}