/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_pack.h"

#include "ccu_log.h"
#include "hcom_common.h"

namespace hcomm {

CcuResPack::~CcuResPack()
{
    if (resHandle_ == 0) {
        return;
    }

    auto ret = CcuReleaseResHandle(devLogicId_, resHandle_);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuResPack][%s] failed, resHandle[%p] devLogicId[%d].",
            __func__, resHandle_, devLogicId_);
    }
    resHandle_ = 0;
}

CcuResult CcuResPack::Init()
{
    devLogicId_ = HcclGetThreadDeviceId();
    if (insType_ == CcuInstanceType::CCU_UNUSED) {
        HCCL_ERROR("[CcuResPack][%s] failed, error ccu instance type[%d].",
            __func__, static_cast<int32_t>(insType_));
        return CcuResult::CCU_E_PARA;
    }

    // 根据通信域算子展开模式申请资源
    // 如果资源不足，返回HCCL_E_UNAVAIL，表示需要回退
    auto ret = CcuAllocResHandleByInsType(devLogicId_, insType_, resHandle_);
    if (ret == CcuResult::CCU_E_UNAVAIL) {
        HCCL_RUN_WARNING("[%s] failed but passed, resource is not enough, "
            "devLogicId[%d], ccuInsType[%d].", __func__, devLogicId_, insType_);
        return ret;
    }
    CCU_CHK_RET(ret);
    CCU_CHK_RET(Reset());
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResPack::Reset()
{
    if (!resHandle_) {
        return CcuResult::CCU_SUCCESS;
    }

    CCU_CHK_RET(CcuCheckResource(devLogicId_, resHandle_, resRepo_));
    return CcuResult::CCU_SUCCESS;
}

CcuResRepository &CcuResPack::GetCcuResRepo()
{
    return resRepo_;
}

} // namespace hcomm
