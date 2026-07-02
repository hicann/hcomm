/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_COMM_DFX_LITE_H
#define HCCL_COMM_DFX_LITE_H
#include "mirror_task_manager_lite.h"
#include "hcclCommProfilingLite.h"
#include "read_write_lock.h"
#include "hccl_common.h"
#include "buffer.h"
#include "common.h"
#include "hcclCommOp.h"

namespace hccl {
// NOTE: HcclCommDfxLite is designed for AICPU single-threaded environments.
// channelRemoteRankIdLite_ and its access methods (AddChannelRemoteRankId/GetChannelRemoteRankId)
// are NOT thread-safe. If used in multi-threaded environments, external synchronization is required.
class HcclCommDfxLite {
public:
     // 构造函数（接收CommunicatorImplLite中已经存在的MirrorTaskManager指针）
    explicit HcclCommDfxLite();

    // 初始化DFX系统 - 修改为返回HcclResult类型
    HcclResult Init(u32 deviceId, const std::string& comTag, u32 rankSize);
    // 获取MirrorTaskManager
    Hccl::MirrorTaskManagerLite* GetMirrorTaskManagerLite() const;

    // Profiling相关接口（直接暴露，不通过GetProfilingImpl）- 全部修改为返回HcclResult类型
    HcclResult ReportAllTasks();
    HcclResult ReportHcclOpInfo(const Hccl::DfxOpInfo& hcclOpInfo);
    HcclResult UpdateProfStat();
    HcclResult SetCurrDfxOpInfo(std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfo);
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> GetCallback() {
        return addTaskCallback_;
    }
    // 将remoteRankId添加到channelRemoteRankId_表中
    void AddChannelRemoteRankId(u64 handle, u32 remoteRankId);
    // 在channelRemoteRankId_表中对remoteRankId进行查找
    u32 GetChannelRemoteRankId(u64 handle);
private:
    std::unique_ptr<Hccl::MirrorTaskManagerLite> mirrorTaskManagerLite_;
    std::unique_ptr<HcclCommProfilingLite> profilingImpl_;
    std::unordered_map<u64, u32> channelRemoteRankIdLite_;
    std::string commTag_;
    u32 deviceId_;
    u32 rankSize_{0};
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> addTaskCallback_;
    bool initializedFlag_{false};
};

}
#endif