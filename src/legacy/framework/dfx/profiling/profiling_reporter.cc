/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "profiling_reporter.h"
#include "dlprof_function.h"
#include "communicator_impl.h"
namespace Hccl {
std::array<ProfilingReporter::lastPosesMap, MAX_MODULE_DEVICE_NUM> ProfilingReporter::allLastPoses_{};
ProfilingReporter::ProfilingReporter(MirrorTaskManager *mirrorTaskMgr, ProfilingHandler* profilingHandler) 
{
    HCCL_INFO("[ProfilingReporter]ProfilingReporter Construct start.");
    if (mirrorTaskMgr == nullptr || profilingHandler == nullptr) {
        THROW<InternalException>("[ProfilingReporter] mirrorTaskMgr or profilingHandler is nullptr.");
    }
    profilingHandler_ = profilingHandler;
    mirrorTaskMgr_ = mirrorTaskMgr;
    mirrorTaskMgr_->RegFullyCallBack([this]() { ReportCallBackAllTasks(); });
    HCCL_INFO("[ProfilingReporter]ProfilingReporter Construct end.");
}

ProfilingReporter::~ProfilingReporter()
{
}

void ProfilingReporter::Init() const
{
}

void ProfilingReporter::ReportOp(uint64_t beginTime, bool cachedReq, bool opbased) const
{
    HCCL_INFO("[ProfilingReporter]ProfilingReporter reportOp start.");
    std::shared_ptr<DfxOpInfo> opInfo = mirrorTaskMgr_->GetCurrDfxOpInfo();
    if (opInfo == nullptr) {
        THROW<InternalException>("[ProfilingReporter]ProfilingReporter reportOp failed, opInfo is nullptr.");
        return;
    }
    uint64_t endTime   = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    OpType   opType    = opInfo->op_.opType;
    bool isAiCpu = false;
    // 新老流程判断
    if (opInfo->isIndop_ == true) {
        if (opInfo->engine == COMM_ENGINE_AICPU_TS || opInfo->engine == COMM_ENGINE_AICPU) {
            HCCL_INFO("[ProfilingReporter][ReportOp] ReportOp Aicpu,opInfo->engine:[%d]", opInfo->engine);
            isAiCpu = true;
        }
    } else {
        CommunicatorImpl *commImp = static_cast<CommunicatorImpl *>(opInfo->comm_);
        CHECK_NULLPTR(commImp, "[]commImp is nullptr!");
        isAiCpu = commImp->GetOpAiCpuTSFeatureFlag();
    }
    // 上报op信息
    opInfo->endTime_ = endTime;
    profilingHandler_->ReportHcclOp(*opInfo, cachedReq);
    
    // 单算子模式涉及HOST API信息上报 注意这个地方
    if (opbased) {
        profilingHandler_->ReportHostApi(opType, beginTime, endTime, !opbased, isAiCpu);
    }
    HCCL_INFO("[ProfilingReporter]ProfilingReporter reportOp end.");
}

void ProfilingReporter::ReportCallBackAllTasks(bool cachedReq)
{
    HCCL_INFO("[ProfilingReporter]ProfilingReporter ReportCallBackAllTasks start.");
    ReportAllTasks(cachedReq);
}

void ProfilingReporter::ReportAllTasks(bool cachedReq) const
{
    HCCL_INFO("[ProfilingReporter]ProfilingReporter ReportAllTasks start.");
    std::lock_guard<std::mutex> lock(mirrorTaskMgr_->GetTaskMutex());
    s32 deviceLogicId = HrtGetDevice();
    if (deviceLogicId >= static_cast<s32>(MAX_MODULE_DEVICE_NUM) || deviceLogicId < 0) {
        HCCL_ERROR("[ProfilingReporter][ReportAllTasks] deviceLogicId[%d] out of range", deviceLogicId);
        return;
    }
    auto& curLastPoses = allLastPoses_[deviceLogicId];
    if (mirrorTaskMgr_ == nullptr || profilingHandler_ == nullptr) {
        HCCL_ERROR("[ProfilingReporter][ReportAllTasks] mirrorTaskMgr_[%p] or profilingHandler_[%p] is nullptr", mirrorTaskMgr_, profilingHandler_);
        return;
    }
    for (auto it = mirrorTaskMgr_->Begin(); it != mirrorTaskMgr_->End(); ++it) {
        u32  streamId     = it->first;
        Queue<std::shared_ptr<TaskInfo>> *currQueue = it->second;
        if (currQueue == nullptr || currQueue->Begin() == nullptr || currQueue->Tail() == nullptr) {
            HCCL_WARNING("[ProfilingReporter][ReportAllTasks] currQueue is nullptr, continue to next task.");
            continue;
        }
        if (*(*(currQueue->Begin())) == nullptr) {
            HCCL_WARNING("[ProfilingReporter][ReportAllTasks] (*(*(currQueue->Begin())) is nullptr, continue to next task.");
            continue;
        }
        if (curLastPoses.find(streamId) == curLastPoses.end() && currQueue->Begin() != nullptr) { // 是首个任务
            TaskInfo task = (*(*(*currQueue->Begin())));
            HCCL_INFO("[ProfilingReporter] ReportTask, streamId = %u, taskId = %u", task.streamId_, task.taskId_);
            profilingHandler_->ReportHcclTaskApi(task.taskParam_.taskType, task.taskParam_.beginTime,
                                                 task.taskParam_.endTime, task.isMaster_, cachedReq, true);
            profilingHandler_->ReportHcclTaskDetails(task, cachedReq);
            curLastPoses[streamId] = currQueue->Begin();
        }
        
        auto endPos = currQueue->Tail();
        auto iter = curLastPoses[streamId];
        ++(*(iter));
        for (; (*(iter)) != (*(currQueue->End())); ++(*(iter))) {
            TaskInfo task = (*(*(*iter)));
            HCCL_INFO("[ProfilingReporter] ReportTask, streamId = %u, taskId = %u", task.streamId_, task.taskId_);
            profilingHandler_->ReportHcclTaskApi(task.taskParam_.taskType, task.taskParam_.beginTime,
                                                 task.taskParam_.endTime, task.isMaster_, cachedReq, true);
            profilingHandler_->ReportHcclTaskDetails(task, cachedReq);
        }
        curLastPoses[streamId] = endPos;
    }

    HCCL_INFO("[ProfilingReporter]ProfilingReporter ReportAllTasks end.");
}

/* 中途打开profiling开关 */
void ProfilingReporter::UpdateProfStat(void)
{
    if (enableHcclL1_ == true) {
        return;
    }
    HCCL_INFO("[ProfilingReporter]ProfilingReporter UpdateProfStat start.");
    // 读取L1开关状态，更新reporter中的开关；
    bool newEnableHcclL1 = profilingHandler_->GetHcclL1State();
    if (mirrorTaskMgr_ == nullptr) {
        THROW<InternalException>("[ProfilingReporter]UpdateProfStat failed, mirrorTaskMgr_ is nullptr.");
    }
    if (enableHcclL1_ != newEnableHcclL1) {
        enableHcclL1_ = newEnableHcclL1;
        s32 deviceLogicId = HrtGetDevice();
        if (deviceLogicId >= static_cast<s32>(MAX_MODULE_DEVICE_NUM) || deviceLogicId < 0) {
            HCCL_ERROR("[ProfilingReporter][ReportAllTasks] deviceLogicId[%d] out of range", deviceLogicId);
            return;
        }
        auto& curLastPoses = allLastPoses_[deviceLogicId];
        for (auto it = mirrorTaskMgr_->Begin(); it != mirrorTaskMgr_->End(); ++it) {
            u32 streamId = it->first;
            if (it->second == nullptr) {
                continue;
            }
            curLastPoses[streamId] = it->second->Tail();
        }
    }
    HCCL_INFO("[ProfilingReporter]ProfilingReporter UpdateProfStat end.");
}

void ProfilingReporter::CallReportMc2CommInfo(const Stream &kfcStream, Stream &stream, const std::vector<Stream *> &aicpuStreams,
                                   const std::string &id, RankId myRank, u32 rankSize, RankId rankInParentComm) const
{
    profilingHandler_->ReportHcclMC2CommInfo(kfcStream, stream, aicpuStreams, id, myRank, rankSize, rankInParentComm);
}

void ProfilingReporter::CallReportMc2CommInfo(const u32 kfcStreamId,
                                            const std::vector<u32> &aicpuStreamsId, const std::string &id,
                                            RankId myRank, u32 rankSize, RankId rankInParentComm) const
{
    profilingHandler_->ReportHcclMC2CommInfo(kfcStreamId, aicpuStreamsId, id,
                                            myRank, rankSize, rankInParentComm);
}
 
} // namespace Hccl