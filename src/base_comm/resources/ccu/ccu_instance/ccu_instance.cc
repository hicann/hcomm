/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_instance.h"

#include <algorithm>

#include "log.h"
#include "ccu_log.h"

#include "hcom_common.h"
#include "exception_handler.h"

#include "ccu_dev_mgr.h"
#include "ccu_res_pack.h"
#include "ccu_kernel_mgr.h"

namespace hcomm {

CcuInstance::~CcuInstance()
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
        ccuDrvHandle_ = nullptr; // 先减少引用计数，再尝试关闭
        (void)CcuDeinitFeature(devLogicId_);
        // 尝试关闭CCU功能，最后一个通信域调用时会关闭CCU驱动
    }
}

CcuResult CcuInstance::Init()
{
    if (insType_ >= CcuInstanceType::CCU_UNUSED) {
        HCCL_ERROR("[CcuInstance][%s] failed, CcuInstanceType[%d] is invalid.",
            __func__, insType_);
        return CcuResult::CCU_E_PARA;
    }

    devLogicId_ = HcclGetThreadDeviceId();

    if (!ccuDrvHandle_) {
        CCU_CHK_RET(CcuInitFeature(devLogicId_, ccuDrvHandle_));
    }

    if (!resPack_) {
        resPack_.reset(new (std::nothrow) CcuResPack(insType_));
        CCU_CHK_PTR_NULL(resPack_);
        CCU_CHK_RET(resPack_->Init());
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::Reset()
{
    if (!resPack_) {
        return CcuResult::CCU_SUCCESS;
    }

    untranslatedKernelHandles_.clear();
    CCU_CHK_RET(resPack_->Reset());
    return CcuResult::CCU_SUCCESS;
}

CcuResPack *CcuInstance::GetResPack()
{
    return resPack_.get();
}

CcuResult CcuInstance::SaveKernel(const CcuKernelHandle kernelHandle)
{
    kernelHandles_.push_back(kernelHandle);
    untranslatedKernelHandles_.push_back(kernelHandle);
    return CcuResult::CCU_SUCCESS;
}

const std::vector<CcuKernelHandle> &CcuInstance::GetUntranslatedKernels()
{
    return untranslatedKernelHandles_;
}

CcuResult CcuInstance::BeginRegister()
{
    if (registerState_ == RegisterState::REGISTERING) {
        HCCL_ERROR("[CcuInstance][%s] failed, previous register round is not ended, "
            "HcommCcuKernelRegisterEnd is missing before a new HcommCcuKernelRegisterStart.", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    registerState_ = RegisterState::REGISTERING;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::CheckRegistering() const
{
    if (registerState_ != RegisterState::REGISTERING) {
        HCCL_ERROR("[CcuInstance][%s] failed, HcommCcuKernelRegister must be called between "
            "HcommCcuKernelRegisterStart and HcommCcuKernelRegisterEnd.", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::EndRegister()
{
    if (registerState_ == RegisterState::IDLE) {
        HCCL_ERROR("[CcuInstance][%s] failed, HcommCcuKernelRegisterEnd is called without a matching "
            "HcommCcuKernelRegisterStart.", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    if (registerState_ == RegisterState::REGISTER_ABORTED) {
        HCCL_WARNING("[CcuInstance][%s] previous register round was aborted due to error, "
            "close it to keep Start/End paired, no kernel will be translated.", __func__);
    }
    registerState_ = RegisterState::IDLE;
    return CcuResult::CCU_SUCCESS;
}

void CcuInstance::AbortRegister()
{
    for (auto kernelHandle : untranslatedKernelHandles_) {
        if (kernelHandle == 0) {
            continue;
        }
        (void)CcuKernelMgr::GetInstance(devLogicId_).UnRegister(kernelHandle);
        auto it = std::find(kernelHandles_.begin(), kernelHandles_.end(), kernelHandle);
        if (it != kernelHandles_.end()) {
            kernelHandles_.erase(it);
        }
    }
    untranslatedKernelHandles_.clear();
    registerState_ = RegisterState::REGISTER_ABORTED;
}

} // namespace hcomm
