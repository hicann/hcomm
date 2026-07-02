/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hcclCommProfilingLite.h"
#include "profiling_reporter_lite.h"
#include "mirror_task_manager_lite.h"

namespace hccl {
// 构造函数
HcclCommProfilingLite::HcclCommProfilingLite(Hccl::DevId deviceId, Hccl::MirrorTaskManagerLite* mirrorTaskManagerLite)
    : mirrorTaskManagerLite_(mirrorTaskManagerLite) { (void)deviceId; }

HcclResult HcclCommProfilingLite::Init() {
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    CHK_RET(Hccl::ProfilingHandlerLite::GetInstance().Init());
    profilingReporterLite_ = std::make_unique<Hccl::ProfilingReporterLite>(
        mirrorTaskManagerLite_, &Hccl::ProfilingHandlerLite::GetInstance(), true);
    CHK_RET(profilingReporterLite_->Init());
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

// HcclCommProfilingLite任务上报
void HcclCommProfilingLite::ReportAllTasks() {
    profilingReporterLite_->ReportAllTasks();
}

void HcclCommProfilingLite::UpdateProfStat() {
    profilingReporterLite_->UpdateProfStat();
}

Hccl::MirrorTaskManagerLite* HcclCommProfilingLite::GetMirrorTaskManagerLite() const {
    return mirrorTaskManagerLite_;
}
}