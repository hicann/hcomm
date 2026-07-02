/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef __AICPU_INDOP_PROCESS_H__
#define __AICPU_INDOP_PROCESS_H__

#include "common.h"
#include "channel_param.h"
#include "aicpu_launch_manager.h"
#include "aicpu_init_param.h"
#include "coll_comm_aicpu_mgr.h"
#include "hccl_diag.h"
class AicpuIndopProcess {
public:
    ~AicpuIndopProcess() = default;
    static HcclResult AicpuIndOpChannelInit(HcclChannelUrmaRes *commParam);
    static HcclResult AicpuIndOpChannelUpdate(HcclChannelUrmaRes *commParam);
    static HcclResult AicpuIndOpThreadInit(ThreadMgrAicpuParam *param);
    static HcclResult AicpuIndOpNotifyInit(NotifyMgrAicpuParam *param);
    static HcclResult AicpuIndOpCommInit(CommAicpuParam *commAicpuParam);
    static HcclResult AicpuDfxOpInfoInit(HcclDfxOpInfo *aicpuDfxInfo, const std::string& commTag);

    static HcclResult AcquireAicpuCommMgr(const std::string &group, CollCommAicpuMgr **aicpuCommMgrPtr);
    static CollCommAicpuMgr *AicpuGetCommMgrbyGroup(const std::string &group);
    static void AicpuReleaseCommMgrbyGroup(const std::string &group);
    static CollCommAicpuMgr *AicpuGetComm(const std::string &group);
    static ReadWriteLockBase& AicpuGetCommMutex();
    static HcclResult AicpuGetCommAll(std::vector<std::pair<std::string, CollCommAicpuMgr *>> &aicpuCommInfo);
    static HcclResult AicpuDestroyCommbyGroup(const std::string &group);

    static HcclResult ProfilingReportDeviceOp();
    static HcclResult UpdateTask(const std::string &group);
};
#endif // __AICPU_INDOP_PROCESS_H__
