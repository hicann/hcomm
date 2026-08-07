/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_aicpu_mgr.h"

HcclResult CollCommAicpuMgr::AcquireCollCommAicpu()
{
    if (collCommAicpu_ != nullptr) {
        HCCL_DEBUG("collCommAicpu_ is not nullptr, no need acquire.");
        return HCCL_SUCCESS;
    }

    EXCEPTION_CATCH(collCommAicpu_ = std::make_unique<CollCommAicpu>(), return HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpuMgr::InitAicpuIndOp(CommAicpuParam* commAicpuParam)
{
    CHK_PTR_NULL(collCommAicpu_);
    return collCommAicpu_->InitAicpuIndOp(commAicpuParam);
}

HcclResult CollCommAicpuMgr::InitThreads(ThreadMgrAicpuParam* param)
{
    CHK_PTR_NULL(collCommAicpu_);
    return collCommAicpu_->InitThreads(param);
}

HcclResult CollCommAicpuMgr::AllocChannelResource(HcclChannelUrmaRes* commParam)
{
    CHK_PTR_NULL(collCommAicpu_);
    return collCommAicpu_->AllocChannelResource(commParam);
}

HcclResult CollCommAicpuMgr::UpdateChannelResource(HcclChannelUrmaRes* commParam)
{
    CHK_PTR_NULL(collCommAicpu_);
    return collCommAicpu_->Resume(commParam);
}

HcclResult CollCommAicpuMgr::NotifyFree(NotifyMgrAicpuParam* param)
{
    CHK_PTR_NULL(collCommAicpu_);
    return collCommAicpu_->NotifyFree(param);
}

HcclResult CollCommAicpuMgr::NotifyAlloc(NotifyMgrAicpuParam* param)
{
    CHK_PTR_NULL(collCommAicpu_);
    return collCommAicpu_->NotifyAlloc(param);
}

bool CollCommAicpuMgr::IsUsed()
{
    std::shared_lock<std::shared_mutex> rwlock(isUsedMutex_);
    return isUsed_;
}

void CollCommAicpuMgr::SetUsed(bool used)
{
    std::unique_lock<std::shared_mutex> rwlock(isUsedMutex_);
    isUsed_ = used;
}

void CollCommAicpuMgr::SetOldA5Comm(void* oldA5Comm)
{
    if (oldA5Comm_ == nullptr) {
        oldA5Comm_ = oldA5Comm;
    }
}
