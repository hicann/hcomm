/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "task_info.h"
#include "log.h"
#include "string_util.h"
#include "const_val.h"
#include "reduce_op.h"

namespace Hccl {
using namespace std;

TaskInfo::TaskInfo(u32 streamId, u32 taskId, u32 remoteRank, const TaskParam& taskParam, const std::shared_ptr<DfxOpInfo>& dfxOpInfo, bool isMaster)
    : streamId_(streamId), taskId_(taskId), remoteRank_(remoteRank), taskParam_(taskParam), dfxOpInfo_(dfxOpInfo), isMaster_(isMaster)
{}

std::string TaskInfo::Describe() const
{
    return StringFormat("TaskInfo[streamId(sqId):[%u], taskId(sqeId):[%u], remoteRank:[%u], taskParam:[%s], dfxOpInfo:[%s], isMaster[%d]]",
                        streamId_, taskId_, remoteRank_, taskParam_.Describe().c_str(),
                        dfxOpInfo_ == nullptr ? "nullptr" : dfxOpInfo_->Describe().c_str(), isMaster_);
}

string TaskInfo::GetAlgTypeName() const
{
    if (this->dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]dfxOpInfo is nullptr, no op info registered, return default value. "
            "streamId(sqId)[%u], taskId(sqeId)[%u].", __func__, streamId_, taskId_);
        return "NULL";
    }
    return this->dfxOpInfo_->algType_;
}

string TaskInfo::GetBaseInfo() const
{
    if (this->dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]dfxOpInfo is nullptr, no op info registered, return default value. "
            "streamId(sqId)[%u], taskId(sqeId)[%u].", __func__, streamId_, taskId_);
        return "";
    }
    return StringFormat("streamID(sqId):[%u], taskID(sqeId):[%u], taskType:[%s], tag:[%s], algType:[%s]",
        this->streamId_,
        this->taskId_,
        this->taskParam_.taskType.Describe().c_str(),
        this->dfxOpInfo_->tag_.c_str(),
        this->GetAlgTypeName().c_str());
}

string TaskInfo::GetParaInfo() const
{
    switch (this->taskParam_.taskType) {
        case TaskParamType::TASK_SDMA:
        case TaskParamType::TASK_RDMA:
        case TaskParamType::TASK_SEND_PAYLOAD:
        case TaskParamType::TASK_UB_INLINE_WRITE:
        case TaskParamType::TASK_UB:
        case TaskParamType::TASK_WRITE_WITH_NOTIFY:
        case TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY:
        case TaskParamType::TASK_DPU_INLINE_WRITE:
        case TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY:
            return GetParaDMA();
        case TaskParamType::TASK_REDUCE_INLINE:
        case TaskParamType::TASK_UB_REDUCE_INLINE:
        case TaskParamType::TASK_REDUCE_TBE:
            return GetParaReduce();
        case TaskParamType::TASK_NOTIFY_RECORD:
        case TaskParamType::TASK_NOTIFY_WAIT:
        case TaskParamType::TASK_SEND_NOTIFY:
        case TaskParamType::TASK_DPU_NOTIFY_WAIT:
        case TaskParamType::TASK_DPU_CHANNEL_FENCE:
            return GetParaNotify();
        case TaskParamType::TASK_AIV:
            return GetParaAiv();
        default:
            return this->taskParam_.taskType.Describe();
    }
}

string TaskInfo::GetParaDMA() const
{
    const auto& taskPara = this->taskParam_.taskPara;
    return StringFormat("src:[0x%llx], dst:[0x%llx], size:[0x%llx], notify id:[0x%016llx], "
                        "link type:[%s], remote rank:[%s]",
                        static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src)),
                        static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst)),
                        static_cast<u64>(taskPara.DMA.size),
                        taskPara.DMA.notifyID,
                        taskPara.DMA.linkType.Describe().c_str(),
                        this->GetRemoteRankInfo().c_str());
}

string TaskInfo::GetParaReduce() const
{
    const auto& taskPara = this->taskParam_.taskPara;
    return StringFormat("src:[0x%llx], dst:[0x%llx], size:[0x%llx], notify id:[0x%016llx], "
                        "op:[%u], data type:[%u], link type:[%s], remote rank:[%s]",
                        static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.src)),
                        static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.dst)),
                        static_cast<u64>(taskPara.Reduce.size),
                        taskPara.Reduce.notifyID,
                        static_cast<u32>(taskPara.Reduce.reduceOp),
                        static_cast<u32>(taskPara.Reduce.dataType),
                        taskPara.Reduce.linkType.Describe().c_str(),
                        this->GetRemoteRankInfo().c_str());
}

string TaskInfo::GetParaNotify() const
{
    const auto& taskPara = this->taskParam_.taskPara;
    return StringFormat("notify id:[0x%016llx], value:[%u], remote rank:[%s]",
        taskPara.Notify.notifyID,
        taskPara.Notify.value,
        this->GetRemoteRankInfo().c_str());
}

string TaskInfo::GetOpInfo() const
{
    if (this->dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]dfxOpInfo is nullptr, no op info registered, return default value. "
            "streamId(sqId)[%u], taskId(sqeId)[%u].", __func__, streamId_, taskId_);
        return "";
    }
    const auto opInfo = this->dfxOpInfo_;
    return StringFormat("commIndex[%u], opType[%s], commId[%s], count[%llu], reduceType[%s], dataType[%s]",
        opInfo->commIndex_,
        opInfo->op_.opType.Describe().c_str(),
        opInfo->commId_.c_str(),
        opInfo->op_.dataCount,
        opInfo->op_.reduceOp.Describe().c_str(),
        opInfo->op_.dataType.Describe().c_str());
}

string TaskInfo::GetRemoteRankInfo(bool needConcise) const
{
    string invRank = needConcise ? "/" : "local";
    return (this->GetRemoteRankId() == INVALID_VALUE_RANKID) ? invRank : to_string(this->GetRemoteRankId());
}

string TaskInfo::GetTaskConciseName() const
{
    static const map<TaskParamType, string> taskConciseNameMap {
            {TaskParamType::TASK_SDMA, "M"},
            {TaskParamType::TASK_RDMA, "RS"},
            {TaskParamType::TASK_SEND_PAYLOAD, "SP"},
            {TaskParamType::TASK_REDUCE_INLINE, "IR"},
            {TaskParamType::TASK_UB_REDUCE_INLINE, "IR"},
            {TaskParamType::TASK_UB, "WorR"},
            {TaskParamType::TASK_REDUCE_TBE, "R"},
            {TaskParamType::TASK_NOTIFY_RECORD, "NR"},
            {TaskParamType::TASK_NOTIFY_WAIT, "NW"},
            {TaskParamType::TASK_SEND_NOTIFY, "SN"},
            {TaskParamType::TASK_WRITE_WITH_NOTIFY, "WN"},
            {TaskParamType::TASK_UB_INLINE_WRITE, "IW"},
            {TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY, "WRN"},
            {TaskParamType::TASK_CCU, "CCU"},
            {TaskParamType::TASK_AICPU_KERNEL, "AIK"}};

    const auto taskName = taskConciseNameMap.find(this->taskParam_.taskType);
    if (taskName == taskConciseNameMap.end()) {
        return "UNKNOWN";
    } else {
        return taskName->second;
    }
}

string TaskInfo::GetNotifyInfo() const
{
    const auto& taskPara = this->taskParam_.taskPara;
    u64 notifyInfo = INVALID_U64;
    switch (this->taskParam_.taskType) {
        case TaskParamType::TASK_RDMA:
        case TaskParamType::TASK_UB_INLINE_WRITE:
            notifyInfo = taskPara.DMA.notifyID;
            break;
        case TaskParamType::TASK_NOTIFY_RECORD:
        case TaskParamType::TASK_NOTIFY_WAIT:
        case TaskParamType::TASK_SEND_NOTIFY:
        case TaskParamType::TASK_WRITE_WITH_NOTIFY:
        case TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY:
            notifyInfo = taskPara.Notify.notifyID;
            break;
        default:
            return "/";
    }
    if (notifyInfo == INVALID_U64) {
        return "/";
    } else {
        stringstream paraStr;
        paraStr << notifyInfo;
        return paraStr.str();
    }
}

string TaskInfo::GetConciseBaseInfo() const
{
    stringstream taskConciseInfo;
    taskConciseInfo << this->GetTaskConciseName();
    taskConciseInfo << "(";
    taskConciseInfo << this->GetRemoteRankInfo(true);
    const auto taskType = this->taskParam_.taskType;
    if (taskType == TaskParamType::TASK_RDMA || taskType == TaskParamType::TASK_NOTIFY_RECORD ||
        taskType == TaskParamType::TASK_NOTIFY_WAIT || taskType == TaskParamType::TASK_SEND_NOTIFY ||
        taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY || taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY ||
 	    taskType == TaskParamType::TASK_UB_INLINE_WRITE) {
        taskConciseInfo << "," << this->GetNotifyInfo();
    }
    taskConciseInfo << ")";
    return taskConciseInfo.str();
}

string TaskInfo::GetIndopBaseInfo() const
{
    return Hccl::StringFormat("streamID(sqId):[%u], taskID(sqeId):[%u], taskType:[%s]",
        this->streamId_, this->taskId_, this->taskParam_.taskType.Describe().c_str());
}

string TaskInfo::GetIndopDataInfo() const
{
    if (this->dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]dfxOpInfo is nullptr, no op info registered, return default value. "
            "streamId(sqId)[%u], taskId(sqeId)[%u].", __func__, streamId_, taskId_);
        return "";
    }

    const auto &opInfo = this->dfxOpInfo_;
    return Hccl::StringFormat("opIndex[%u], algTag[%s], count[%llu], reduceType[%s], dataType[%s], "\
        "input: ptr[0x%llx] size[%llu], output: ptr[0x%llx] size[%llu]",
        opInfo->opIndex_,
        opInfo->algTag_.c_str(),
        opInfo->op_.dataCount,
        opInfo->op_.reduceOp.Describe().c_str(),
        opInfo->op_.dataType.Describe().c_str(),
        opInfo->op_.inputMem == nullptr ? 0 : static_cast<u64>(opInfo->op_.inputMem->GetAddr()),
        opInfo->op_.inputMem == nullptr ? 0 : opInfo->op_.inputMem->GetSize(),
        opInfo->op_.outputMem == nullptr ? 0 : static_cast<u64>(opInfo->op_.outputMem->GetAddr()),
        opInfo->op_.outputMem == nullptr ? 0 : opInfo->op_.outputMem->GetSize());
}

string TaskInfo::GetParaAiv() const
{
    const auto &taskPara = this->taskParam_.taskPara;
    return StringFormat("cmdType:[%d], tag:[%u], count:[%llu], numBlocks:[%u], rankSize:[%u], "
                        "rank:[%u], sendRecvRemoteRank:[%u], dataType:[%d], remote rank:[%s]",
                        static_cast<int>(taskPara.Aiv.cmdType), taskPara.Aiv.tag,
                        taskPara.Aiv.count, taskPara.Aiv.numBlocks, taskPara.Aiv.rankSize,
                        taskPara.Aiv.rank, taskPara.Aiv.sendRecvRemoteRank,
                        static_cast<int>(taskPara.Aiv.dataType),
                        this->GetRemoteRankInfo().c_str());
}

u32 TaskInfo::GetRemoteRankId() const
{
    return getRemoteRankByHandle_ ? getRemoteRankByHandle_(channelHandle_) : remoteRank_;
}

} // namespace Hccl