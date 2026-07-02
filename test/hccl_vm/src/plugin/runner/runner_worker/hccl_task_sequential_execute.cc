/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_task_sequential_execute.h"

#include <cstdint>
#include <iostream>

#include "device_resource_manager.h"
#include "sim_log.h"

using namespace HcclSim;

namespace VirtualRunTime {
SqeuentialExecutor::SqeuentialExecutor(AllRankTaskQueues &allRankTaskQueues)
{
    allRankTaskQueues_ = allRankTaskQueues;
}

const std::map<HccLTaskMetaType, const std::string> SqeuentialExecutor::taskNames_ = {
    {HccLTaskMetaType::REDUCE, "reduce"},
    {HccLTaskMetaType::MEM_CPY, "mem_cpy"},
    {HccLTaskMetaType::NOTIFY_RECORD, "record"},
    {HccLTaskMetaType::CCU_GRAPH, "ccu_graph"},
    {HccLTaskMetaType::AIV_GRAPH, "aiv_graph"},
    {HccLTaskMetaType::NOTIFY_WAIT, "wait"}};

HcclVmResult SqeuentialExecutor::Execute()
{
    auto rankSize = allRankTaskQueues_.size();
    auto &devResMgr = DeviceResourceManager::GetInstance();
    devResMgr.Init(rankSize);
    while (HasTask()) {
        uint32_t rankId = 0;
        for (auto& rankTasks : allRankTaskQueues_) { // rank
            devResMgr.InitRankRes(rankId++, rankTasks.size());
            for (auto& streamTasks : rankTasks) { // stream
                while (!streamTasks.empty()) {
                    auto task = streamTasks.front();
                    auto ret = ExecuteOneTask(task);
                    if (ret == HcclVmResult::HCCL_SIM_VRT_HOLD_CMD) {
                        break;
                    } else if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
                        HCCL_VM_ERROR("ExecuteOneTask failed, ret: {}, rankId = {}, type= {}",
                            static_cast<int>(ret), rankId, taskNames_.at(task.taskType));
                        return ret;
                    }
                    streamTasks.pop();
                }
            }
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult SqeuentialExecutor::ExecuteOneTask(HcclTaskMetaData& task)
{
    switch (task.taskType) {
        case HccLTaskMetaType::REDUCE:
            return TaskReduce(task);
        case HccLTaskMetaType::MEM_CPY:
            return TaskMemcpy(task);
        case HccLTaskMetaType::NOTIFY_RECORD:
            return TaskNotifyRecord(task);
        case HccLTaskMetaType::NOTIFY_WAIT:
            return TaskNotifyWait(task);
        case HccLTaskMetaType::CCU_GRAPH:
            return TaskCcuGraph(task);
        case HccLTaskMetaType::AIV_GRAPH:
            return TaskAivGraph(task);
        default:
            break;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

bool SqeuentialExecutor::HasTask()
{
    for (auto& rankTasks : allRankTaskQueues_) {
        for (auto& streamTasks : rankTasks) {
            if (!streamTasks.empty()) {
                return true;
            }
        }
    }
    return false;
}
}
