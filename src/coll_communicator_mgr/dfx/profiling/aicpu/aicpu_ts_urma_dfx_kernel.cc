/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "aicpu_indop_process.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuDfxOpInfoInitV2(void* args)
{
    HCCL_RUN_INFO("RunAicpuDfxOpInfoInitV2 start.");
    CHK_PRT_RET(args == nullptr, HCCL_ERROR("[%s]args is null.", __func__), HCCL_E_PARA);
    struct InitTask {
        u64 context;
        char commTag[256];
    };
    InitTask* ctxArgs = reinterpret_cast<InitTask*>(args);
    CHK_PRT_RET(ctxArgs == nullptr, HCCL_ERROR("[%s]ctxArgs is null.", __func__), HCCL_E_PARA);
    HcclDfxOpInfo* dfxOpInfo = reinterpret_cast<HcclDfxOpInfo*>(ctxArgs->context);
    return AicpuIndopProcess::AicpuDfxOpInfoInit(dfxOpInfo, ctxArgs->commTag);
}
}
