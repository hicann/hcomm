/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hcclCommDfx.h"

namespace hccl {

ReadWriteLockBase HcclCommDfx::baseLock_;
ReadWriteLock HcclCommDfx::rwLock_(HcclCommDfx::baseLock_);
std::unordered_map<std::string,std::unordered_map<u64, u32> > HcclCommDfx::channelRemoteRankId_;
std::unordered_map<u32, u32> HcclCommDfx::streamIdToTaskId_;
HcclCommDfx::HcclCommDfx() {
}

HcclCommDfx::~HcclCommDfx() {
    setAddTaskCallback_ = nullptr;
    setAddDpuTaskCallback_ = nullptr;
}

HcclResult HcclCommDfx::Init(u32 deviceId, const std::string& comTag, u32 myRankId) {
    HCCL_INFO("[%s]deviceId[%u], comTag[%s], myRankId[%u]", __func__, deviceId, comTag.c_str(), myRankId);
    deviceId_ = deviceId;
    commTag_ = comTag;
    myRankId_ = myRankId;
    // 1. 如果mirrorTaskManager_为空，则创建新的MirrorTaskManager
    if (!mirrorTaskManager_) {
        mirrorTaskManager_ = std::make_unique<Hccl::MirrorTaskManager>(deviceId_, &Hccl::GlobalMirrorTasks::Instance(), false);
    }
    
    // 2. 创建Profiling管理类
    EXECEPTION_CATCH(profiling_ = std::make_unique<HcclCommProfiling>(deviceId_, mirrorTaskManager_.get()), return HCCL_E_PTR);
    
    // 3. 注册回调
    setAddTaskCallback_ = [this](u32 streamId, u32 taskId, const Hccl::TaskParam &taskParam, u64 handle) {
        return this->AddTaskInfoCallback(streamId, taskId, taskParam, handle);
    };
    setAddDpuTaskCallback_ = [this](const Hccl::TaskParam &taskParam, u64 handle) {
        return this->AddDpuTaskInfoCallback(taskParam, handle);
    };
    HCCL_INFO("[HcclCommDfx][Init] Init success");
    return HCCL_SUCCESS; // 初始化成功返回成功码
}

HcclResult HcclCommDfx::IsOpBase(bool &isOpBase) {
    CHK_SMART_PTR_NULL(mirrorTaskManager_);
    auto currDfxOpInfo = mirrorTaskManager_->GetCurrDfxOpInfo();
    CHK_SMART_PTR_NULL(currDfxOpInfo);
    isOpBase = currDfxOpInfo->op_.opMode == Hccl::OpMode::OPBASE;
    HCCL_INFO("[%s] IsOpBase: %d", __func__, isOpBase);
    return HCCL_SUCCESS;
}


// 回调注册实现
HcclResult HcclCommDfx::AddTaskInfoCallback(u32 streamId, u32 taskId, const Hccl::TaskParam &taskParam, u64 handle) {
    CHK_SMART_PTR_NULL(mirrorTaskManager_);
    u32 remoteRankId = INVALID_UINT;
    if (handle != INVALID_U64) {
        CHK_RET(GetChannelRemoteRankId(commTag_, handle, remoteRankId));
    }
    std::shared_ptr<Hccl::TaskInfo> taskInfo{nullptr};
    EXECEPTION_CATCH(taskInfo = std::make_shared<Hccl::TaskInfo>(streamId, taskId,
        remoteRankId, taskParam, mirrorTaskManager_->GetCurrDfxOpInfo(), taskParam.isMaster), return HCCL_E_PTR);
    EXECEPTION_CATCH(mirrorTaskManager_->AddTaskInfo(taskInfo), return HCCL_E_PTR);
    HCCL_INFO("[%s]taskInfo: %s", __func__, taskInfo->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommDfx::AddDpuTaskInfoCallback(const Hccl::TaskParam &taskParam, u64 handle) {
    u32 streamId = dpuStreamId_;
    u32 taskId = GetTaskId(streamId);
    Hccl::TaskParam localTaskParam = taskParam;
    localTaskParam.aicpuTaskId = aicpuTaskId_;
    localTaskParam.npuDevId = deviceId_;
    HCCL_INFO("[%s] streamId[%u], taskId[%u], aicpuTaskId[%u], npuDevId[%u].", __func__, streamId, taskId, localTaskParam.aicpuTaskId, localTaskParam.npuDevId);
    return AddTaskInfoCallback(streamId, taskId, localTaskParam, handle);
}

// HcclCommDfx接口实现 - 修改为返回HcclResult类型
HcclResult HcclCommDfx::ReportAllTasks(bool cachedReq) {
    CHK_PTR_NULL(profiling_);
    EXECEPTION_CATCH(profiling_->ReportAllTasks(cachedReq), return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

HcclResult HcclCommDfx::ReportOp(u64 beginTime, bool cachedReq, bool opbased) {
    CHK_PTR_NULL(profiling_);
    EXECEPTION_CATCH(profiling_->ReportOp(beginTime, cachedReq, opbased), return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

// 返回值Mc2要改
void HcclCommDfx::ReportMc2CommInfo(const Mc2CommInfo& mc2CommInfo) {
    if (profiling_) {
        profiling_->ReportMc2CommInfo(mc2CommInfo);
    }
}

HcclResult HcclCommDfx::UpdateProfStat() {
    CHK_PTR_NULL(profiling_);
    profiling_->UpdateProfStat();
    return HCCL_SUCCESS;
}

Hccl::MirrorTaskManager* HcclCommDfx::GetMirrorTaskManager() const {
    return mirrorTaskManager_.get();
}

// 将remoteRankId添加到channelRemoteRankId_表中
void HcclCommDfx::AddChannelRemoteRankId(const std::string& commTag, u64 handle, u32 remoteRankId) {
    rwLock_.writeLock();
    HCCL_INFO("[HcclCommDfx][AddChannelRemoteRankId] commTag:[%s], handle:[%lu], remoteRankId:[%u]", commTag.c_str(), handle, remoteRankId);
    channelRemoteRankId_[commTag][handle] = remoteRankId;
    rwLock_.writeUnlock();
}

// 在channelRemoteRankId_表中对remoteRankId进行查找（原有逻辑补充返回值）
HcclResult HcclCommDfx::GetChannelRemoteRankId(const std::string& commTag, u64 handle, u32& remoteRankId) {
    rwLock_.readLock();
    auto commIt = channelRemoteRankId_.find(commTag);
    if (commIt == channelRemoteRankId_.end()) {
        rwLock_.readUnlock();
        HCCL_ERROR("[HcclCommDfx]commTag:[%s] not found", commTag.c_str());
        return HCCL_E_PARA;
    }
    auto handleIt = commIt->second.find(handle);
    if (handleIt == commIt->second.end()) {
        HCCL_ERROR("[HcclCommDfx]handle not found,commTag:[%s],handle:[%lu]", commTag.c_str(), handle);
        rwLock_.readUnlock();
        return HCCL_E_PARA;
    }
    remoteRankId = handleIt->second;
    rwLock_.readUnlock();
    return HCCL_SUCCESS; // 查找成功补充返回成功码
}

HcclResult HcclCommDfx::ReportKernel(uint64_t beginTime, const std::string& commTag, const std::string& kernelName, uint32_t threadId, bool cachedReq) {
    CHK_PTR_NULL(profiling_);
    CHK_RET(profiling_->ReportKernel(beginTime, commTag, kernelName, threadId, cachedReq));
    return HCCL_SUCCESS; 
}

constexpr u32 TASKID_RANGE = 65536;
u32 HcclCommDfx::GetTaskId(u32 streamId)
{
    rwLock_.writeLock();
    auto& taskIdRef = streamIdToTaskId_[streamId];
    taskIdRef = (taskIdRef + 1) % TASKID_RANGE;
    u32 retTaskId = taskIdRef; 
    rwLock_.writeUnlock();
    return retTaskId; 
}

void HcclCommDfx::SetDpuStreamId(u32 dpuStreamId) {
    dpuStreamId_ = dpuStreamId;
}

}
