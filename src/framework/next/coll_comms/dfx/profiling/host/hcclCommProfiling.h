/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_COMM_PROFILING_H
#define HCCL_COMM_PROFILING_H

#include <memory>
#include "mirror_task_manager.h"
#include "profiling_reporter.h"
#include "types.h"

namespace hccl {
    struct Mc2CommInfo {
        u32 FreeStreamId;
        std::vector<u32> streamsId;
        std::string groupname;
        u32 myRankId;
        u32 rankSize;
        u32 parentRankId;
    };
    
class HcclCommProfiling {
public:
    // 构造函数
    HcclCommProfiling(u32  deviceId, Hccl::MirrorTaskManager* mirrorTaskManager);
    
    // 上报所有任务
    void ReportAllTasks(bool cachedReq = false);
    
    // 上报算子信息
    void ReportOp(uint64_t beginTime, bool cachedReq, bool opbased);
    
    // 上报MC2通信信息
    void ReportMc2CommInfo(const Mc2CommInfo& mc2CommInfo);
    
    // 更新Profiling统计
    void UpdateProfStat();
    
    // 获取MirrorTaskManager
    Hccl::MirrorTaskManager* GetMirrorTaskManager() const;
    HcclResult ReportKernel(uint64_t beginTime, const std::string& commTag, const std::string& kernelName, uint32_t threadId, bool cachedReq) const;
    
private:
    Hccl::MirrorTaskManager* mirrorTaskManager_;
    std::unique_ptr<Hccl::ProfilingReporter> profilingReporter_;
};
}// namespace hccl
#endif // HCCL_COMM_PROFILING_H