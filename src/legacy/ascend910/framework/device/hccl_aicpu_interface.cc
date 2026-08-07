/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_aicpu_interface.h"

#include <sstream>
#include <dlfcn.h>
#include "common/aicpu_hccl_common.h"
#include "common/aicpu_hccl_def.h"
#include "common/aicpu_sqe_context.h"
#include "profiling_manager_device.h"
#include "framework/aicpu_hccl_process.h"
#include "utils/hccl_aicpu_utils.h"
#include "framework/aicpu_communicator.h"
#include "utils/aicpu_hdc_utils.h"
#include "hccl_group_utils.h"
#include "hccl_dl.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInitV2(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }

    KFCResInitTask* ctxArgs = reinterpret_cast<KFCResInitTask*>(args);
    HCCL_INFO("RunAicpuKfcResInitV2 isCustom %u, context %#llx", ctxArgs->isCustom, ctxArgs->context);
    if (ctxArgs->context == 0) { // for OneSideComm
        CHK_RET(hrtSetWorkModeAicpu(true));
        HCCL_INFO("RunAicpuKfcResInitV2 done as context is null, set aicpu work mode");
        return HCCL_SUCCESS;
    }
    return AicpuHcclProcess::AicpuRpcResInitV2(reinterpret_cast<HcclOpResParam*>(ctxArgs->context), ctxArgs->isCustom);
}

__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunchV2(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 args is null.");
        return HCCL_E_PARA;
    }

    KFCTaskComm* task = reinterpret_cast<KFCTaskComm*>(args);
    OpTilingData* tilingData = reinterpret_cast<OpTilingData*>(reinterpret_cast<std::uintptr_t>(task) + sizeof(u64));
    if (tilingData == nullptr) {
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 tilingData args is null.");
        return HCCL_E_PARA;
    }

    const HcclCMDType opType = static_cast<HcclCMDType>(tilingData->opType);
    if ((opType == HcclCMDType::HCCL_CMD_BATCH_GET) || (opType == HcclCMDType::HCCL_CMD_BATCH_PUT)) {
        return AicpuHcclProcess::HandleOneSideService(tilingData);
    }

    HcclOpResParam* commParam = reinterpret_cast<HcclOpResParam*>(task->context);
    if (commParam == nullptr) {
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 context args is null.");
        return HCCL_E_PARA;
    }
    HCCL_INFO("RunAicpuRpcSrvLaunchV2 KFCTask task %p, context %p, tilingData %p", task, commParam, tilingData);

    std::string group = commParam->hcomId;
    hccl::HcclCommAicpu* hcclCommAicpu = AicpuHcclProcess::AicpuGetCommbyGroup(group);
    if (hcclCommAicpu == nullptr) {
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 get Hcclcomm error group[%s], tag[%s]", commParam->hcomId, tilingData->tag);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[RunAicpuRpcSrvLaunchV2] isZeroCopy [%d], isSymmetricMemory [%d], workflowMode[%d]", tilingData->isZeroCopy,
        tilingData->isSymmetricMemory, tilingData->workflowMode);
    hcclCommAicpu->SetZeroCopyEnable(tilingData->isZeroCopy);
    hcclCommAicpu->SetSymmetricMemoryEnable(tilingData->isSymmetricMemory);
    DfxExtendInfo* dfxInfo = hcclCommAicpu->GetDfxExtendInfo();
    if ((dfxInfo->cqeStatus != dfx::CqeStatus::kDefault) || (dfxInfo->pollStatus == PollStatus::kStopAsException)) {
        AicpuHcclProcess::AicpuReleaseCommbyGroup(group);
        HCCL_ERROR(
            "RunAicpuRpcSrvLaunchV2 exist errors before, cqeStatus:%d, pollStatus:%d, group[%s], sqeType[%u]",
            dfxInfo->cqeStatus, dfxInfo->pollStatus, commParam->hcomId, dfxInfo->cqeException.sqeType);
        if (dfxInfo->cqeException.sqeType == RT_STARS_SQE_TYPE_SDMA) {
            return TS_ERROR_AICPU_SDMA;
        }
        return HCCL_E_INTERNAL;
    }
    SetWorkflowMode(static_cast<HcclWorkflowMode>(tilingData->workflowMode));
    HCCL_DEBUG("[NsRecovery]check the suspending status in hcclCommAicpu");
    HcclComSuspendingFlag kfcFlag = HcclComSuspendingFlag ::isResume;
    CHK_RET(hcclCommAicpu->GetSuspendingFlag(kfcFlag));
    if (kfcFlag == HcclComSuspendingFlag::isSuspending) {
        HCCL_RUN_WARNING("[NsRecovery] the op should not be launched in hcclCommAicpu on the suspending status");
        HCCL_RUN_WARNING("[NsRecovery] Possible reason: flag not refresh, not enough time between resume and launch");
        AicpuHcclProcess::AicpuReleaseCommbyGroup(group);
        return 0;
    }
    hcclCommAicpu->SetNsStopLaunchStatus(false);
    HcclResult res = AicpuHcclProcess::AicpuRunRpcServerV2(hcclCommAicpu, tilingData, commParam);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(group);
    if (res != HCCL_SUCCESS) {
        if (res == HCCL_E_OPRETRY_FAIL) {
            HCCL_RUN_INFO("Retry failed, support step retry");
            return TS_ERROR_RETRY_CONSTRAINT;
        } else if (res != HCCL_E_SUSPENDING) {
            if (dfxInfo->cqeException.sqeType == RT_STARS_SQE_TYPE_SDMA) {
                HCCL_ERROR("run AicpuRunRpcServerV2 failed. ret[%u]", TS_ERROR_AICPU_SDMA);
                return TS_ERROR_AICPU_SDMA;
            }
            HCCL_ERROR("run AicpuRunRpcServerV2 failed. ret[%d]", res);
            return res;
        } else {
            HCCL_INFO("aicpu is suspended");
            return AICPUSUSPENDING_ERROR;
        }
    }
    HCCL_INFO("end RunAicpuRpcSrvLaunchV2");
    return 0;
}

__attribute__((visibility("default"))) uint32_t RunAicpuNotifyRecord(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("RunAicpuNotifyRecord args is null.");
        return HCCL_E_PARA;
    }
    ThreadNotifyRecordParam* param = reinterpret_cast<ThreadNotifyRecordParam*>(args);
    HCCL_INFO(
        "[RunAicpuNotifyRecord] thread[0x%llx], dstThread[0x%llx], dstNotifyIdx[%u]", param->thread, param->dstThread,
        param->dstNotifyIdx);
    int32_t ret = HcommThreadNotifyRecordOnThread(param->thread, param->dstThread, param->dstNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("RunAicpuNotifyRecord failed. ret[%d]", ret);
        return ret;
    }
    HCCL_INFO("RunAicpuNotifyRecord success.");
    return HCCL_SUCCESS;
}

__attribute__((visibility("default"))) uint32_t RunAicpuNotifyWait(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("RunAicpuNotifyWait args is null.");
        return HCCL_E_PARA;
    }
    ThreadNotifyWaitParam* param = reinterpret_cast<ThreadNotifyWaitParam*>(args);
    HCCL_INFO("[RunAicpuNotifyWait] thread[0x%llx], notifyIdx[%u]", param->thread, param->notifyIdx);
    int32_t ret = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(param->thread, param->notifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("RunAicpuNotifyWait failed. ret[%d]", ret);
        return ret;
    }
    HCCL_INFO("RunAicpuNotifyWait success.");
    return HCCL_SUCCESS;
}

} // extern "C"
