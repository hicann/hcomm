/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring> 
#include <algorithm>
#include "profiling_handler.h"
#include "dlprof_function.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sal.h"
#include "task_param.h"
#include "aprof_pub.h"
#include "orion_adapter_rts.h"
#include "communicator_impl.h"
#include "data_type.h"
namespace Hccl {
#define UNUSED(x) (void)(x)

constexpr uint16_t  CCU_TYPE   = 2; //  枚举，0为Task粒度，1为WaitCKE，2为LoopGroup，3为channelId->RemoteRankId的映射

ProfilingHandler ProfilingHandler::instance_;

ProfilingHandler::ProfilingHandler()
{
    Init();
}

ProfilingHandler::~ProfilingHandler()
{
}

ProfilingHandler &ProfilingHandler::GetInstance()
{
    return instance_;
}

void ProfilingHandler::Init()
{
    HCCL_INFO("[ProfilingHandler]Init start.");
    if (initializedFlag_) {
        return;
    }
    if (Hccl::DlProfFunction::GetInstance().DlProfFunctionInit() != HCCL_SUCCESS) {
        THROW<InternalException>("[ProfilingHandler] DlProfFunctionInit failed.");
    }
    // 注册profiling开关状态监测回调函数
    ProfCommandHandle callback = CommandHandleWrapper;
    auto ret                        = DlProfFunction::GetInstance().dlMsprofRegisterCallback(HCCL, callback);
    if (ret != 0) {
        THROW<InternalException>("[ProfilingHandler][Init]errNo[0x%016llx] Prof Register CtrlCallback"
                                 " fail, return[%d]",
                                 HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
    }
    // 注册通信库使用的字符串信息，映射对应hashId
    for (auto i = 0; i < TaskParamType::__COUNT__; ++i) {
        TaskParamType type(static_cast<TaskParamType::Value>(i));
        std::string nameInfo = type.Describe();
        uint64_t    hashId   = GetProfHashId(nameInfo.c_str(), nameInfo.length());
        HCCL_INFO("[TaskParamType] nameInfo[%s] ret[%llu]", nameInfo.c_str(), hashId);
        str2HashId_[nameInfo] = hashId;
    }
    // 保存关键信息
    initializedFlag_ = true;
    HCCL_INFO("[ProfilingHandler]Init end.");
}

// 回调注册
int32_t ProfilingHandler::CommandHandleWrapper(uint32_t rtType, void *data, uint32_t len)
{
    HCCL_INFO("[ProfilingHandler]CommandHandleWrapper start.");
    return instance_.CommandHandle(rtType, data, len);
}

// 接口预留，暂时不实现
// 函数入参，因为静态检查先删除注释：kernelType kerType, uint64_t beginTime, uint64_t endTime, bool cachedReq
void ProfilingHandler::ReportKernel() const
{
}

// 只有单算子走这
void ProfilingHandler::ReportHostApi(OpType opType, uint64_t beginTime, uint64_t endTime, bool cachedReq, bool isAiCpu)
{
    HCCL_INFO("[ProfilingHandler]ReportHostApi start.");
    uint32_t threadId = SalGetTid();
    std::string profName(GetProfOpName(opType));
    if (isAiCpu) {
        profName += "AicpuKernel";
    }
    uint64_t          cmdItemId  = DlProfFunction::GetInstance().dlMsprofStr2Id(profName.c_str(), profName.length());
    if (enableHostApi_) {
        ReportAclApi(opType, beginTime, endTime, cmdItemId, threadId);
    }
    ReportNodeApi(beginTime, endTime, cmdItemId, threadId, cachedReq);
    ReportNodeBasicInfo(endTime, cmdItemId, threadId, cachedReq);
    HCCL_INFO("[ProfilingHandler]ReportHostApi end.");
}

void ProfilingHandler::ReportHcclOp(const DfxOpInfo &opInfo, bool cachedReq)
{
    HCCL_INFO("[ProfilingHandler]ReportHcclOp start.");
    uint32_t threadId = SalGetTid();
    ReportHcclOpInfo(opInfo.endTime_, opInfo, threadId, cachedReq);
    HCCL_INFO("[ProfilingHandler]ReportHcclOp end.");
}

void ProfilingHandler::ReportHcclTaskApi(TaskParamType taskType, uint64_t beginTime, uint64_t endTime, bool isMasterStream, bool cachedReq, bool ignoreLevel)
{
    // 获取数据
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = isMasterStream ? MSPROF_REPORT_HCCL_MASTER_TYPE : MSPROF_REPORT_HCCL_SLAVE_TYPE;
    reporterData.threadId = SalGetTid();
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    const std::string proName(GetProfTaskOpNameV2(taskType));
    reporterData.itemId = GetProfHashId(proName.c_str(), proName.length());
    HCCL_INFO("[ProfilingHandler]ReportHcclTaskApi, reporterData data is: level[%u], type[%u], threadId[%u], "
              "beginTime[%llu], endTime[%llu], itemId[%llu]",
              reporterData.level, reporterData.type, reporterData.threadId, reporterData.beginTime,
              reporterData.endTime, reporterData.itemId);
    // 开关判断，订阅开关未开启时，不上报数据
    if (taskType == TaskParamType::TASK_AICPU_KERNEL) {
        return;
    }
    if (cachedReq) {
        HCCL_INFO("[ProfilingHandler] Cache ReportData");
        std::lock_guard<std::mutex> lock(cachedTaskApiInfoMutex_);
        cachedTaskApiInfo_.push(reporterData);
    }
    if ((!enableHcclNode_) || (!ignoreLevel && !enableHcclL1_)) {
        HCCL_INFO("[ProfilingHandler] ReportHcclTaskApi, enableHcclNode_[%u], enableHcclL1_[%u], ignoreLevel[%u]",
                  enableHcclNode_, enableHcclL1_, ignoreLevel);
        return;
    }
    // 数据上报
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(1, &reporterData); 
    HCCL_INFO("Call MsprofReportApi, return value[%d]", ret);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportApi fail, return[%d]", ret);
    }
    HCCL_INFO("[ProfilingHandler]ReportHcclTaskApi end.");
}

void ProfilingHandler::ReportHcclTaskDetails(const TaskInfo &taskInfo, bool cachedReq)
{
    CHECK_NULLPTR(taskInfo.dfxOpInfo_, "[ProfilingHandler::ReportHcclTaskDetails] taskInfo.dfxOpInfo_ is nullptr!");
    CHECK_NULLPTR(taskInfo.dfxOpInfo_->comm_, 
                  "[ProfilingHandler::ReportHcclTaskDetails] taskInfo.dfxOpInfo_->comm_ is nullptr!");
    if (enableHcclL1_ == false && !cachedReq) {
        return;
    }
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cacheTaskInfosMutex_);
        cacheTaskInfos_.push_back(taskInfo);
        HCCL_INFO("[ProfilingHandler] cache taskInfo.");
    }
    if (enableHcclL1_ == false) {
            return;
    }
    // 数据组装
    HCCL_INFO("[ProfilingHandler]ReportHcclTaskDetails start.");
    HCCLReportData hcclReportData{};
    GetHCCLReportData(taskInfo, hcclReportData);

    // 调用additionInfo接口上报数据
   
    // 上报DPU track信息
    if (taskInfo.taskParam_.taskType == TaskParamType::TASK_DPU_INLINE_WRITE
        || taskInfo.taskParam_.taskType == TaskParamType::TASK_DPU_NOTIFY_WAIT
        || taskInfo.taskParam_.taskType == TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY
        || taskInfo.taskParam_.taskType == TaskParamType::TASK_DPU_CHANNEL_FENCE) {
        HCCL_INFO("[ProfilingHandler]ReportHcclTaskDetails Report DPU info taskId[%llu].", hcclReportData.dpuProfInfo.taskId);
        CallAdditionInfo(hcclReportData, &hcclReportData.dpuProfInfo, sizeof(hcclReportData.dpuProfInfo), ProfTaskType::TASK_DPU_HCCL_INFO);
    } else {
        HCCL_INFO("[ProfilingHandler]ReportHcclTaskDetails Report HCCL info .");
        CallAdditionInfo(hcclReportData, &hcclReportData.profInfo, sizeof(hcclReportData.profInfo), ProfTaskType::TASK_HCCL_INFO);
    }
    HCCL_INFO("[ProfilingHandler]ReportHcclTaskDetails end.");
}

void ProfilingHandler::CallAdditionInfo(HCCLReportData &hcclReportData, void *data, u32 len, ProfTaskType type) const
{
    HCCL_INFO("[ProfilingHandler]ReportHcclTaskDetails start.");
    MsprofAdditionalInfo reporterData{};
    reporterData.level     = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type      = static_cast<uint32_t>(type);
    reporterData.threadId  = SalGetTid();
    reporterData.dataLen   = len;
    reporterData.timeStamp = hcclReportData.ts;
    s32 sret               = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    if (sret != EOK) {
        THROW<InternalException>("Call memcpy_s failed, errorno[%d]", sret);
    }
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(
        1, &reporterData, sizeof(MsprofAdditionalInfo)); // aingFlag 根据静态图模式下保存
    HCCL_INFO("Call MsprofReportAdditionalInfo, return value[%d], taskType[%d]", ret, static_cast<int>(type));
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportAdditionalInfo failed, return[%d]", ret);
    }
    HCCL_INFO("[ProfilingHandler]ReportHcclTaskDetails data is: level[%u], type[%u], threadId[%u],  dataLen[%u], "
              "timeStamp[%llu]",
              reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen,
              reporterData.timeStamp);
    HCCL_INFO("[ProfilingHandler]ReportHcclTaskDetails end.");
}

void ProfilingHandler::GetProfCommonInfo(const TaskInfo &taskInfo, HCCLReportData &hcclReportData) const
{
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[ProfilingHandler]taskInfo.dfxOpInfo_ is nullptr");
        return;
    }
    hcclReportData.ts = taskInfo.taskParam_.endTime;
    const auto &profName = GetProfTaskOpNameV2(taskInfo.taskParam_.taskType);
    hcclReportData.profInfo.itemId = GetProfHashId(profName.c_str(), profName.length());
    const auto &cclTag = taskInfo.dfxOpInfo_->tag_;
    hcclReportData.profInfo.cclTag = GetProfHashId(cclTag.c_str(), cclTag.length());
    const auto &opTag = taskInfo.dfxOpInfo_->op_.opTag;
    uint64_t groupName = GetProfHashId(opTag.c_str(), opTag.length());
    if (taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[ProfilingHandler]taskInfo.dfxOpInfo_->comm_ is nullptr");
        return;
    }
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        hcclReportData.profInfo.groupName = groupName;
        hcclReportData.profInfo.rankSize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl *commImp = static_cast<CommunicatorImpl *>(taskInfo.dfxOpInfo_->comm_);
        if (commImp == nullptr) {
            HCCL_ERROR("[ProfilingHandler]commImp is  nullptr");
            return;
        }
        hcclReportData.profInfo.groupName = groupName;
        hcclReportData.profInfo.rankSize  = commImp->GetRankSize();
    }
    hcclReportData.profInfo.workFlowMode      = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    hcclReportData.profInfo.planeID           = 0;
    hcclReportData.profInfo.stage             = 0;
    hcclReportData.profInfo.role              = static_cast<uint32_t>(TaskRole::DST);
    hcclReportData.profInfo.durationEstimated = 0;
    hcclReportData.profInfo.localRank         = taskInfo.dfxOpInfo_->op_.myRank;
    hcclReportData.profInfo.remoteRank        = taskInfo.remoteRank_;
    hcclReportData.profInfo.dataType          = taskInfo.dfxOpInfo_->op_.dataType;
    hcclReportData.profInfo.opType            = taskInfo.dfxOpInfo_->op_.opType;
    hcclReportData.profInfo.transportType     = static_cast<int32_t>(SimpleTaskType::UB);
    if (hcclReportData.profInfo.remoteRank == INVALID_VALUE_RANKID) {
        hcclReportData.profInfo.transportType = static_cast<int32_t>(SimpleTaskType::LOCAL);
        hcclReportData.profInfo.linkType = 0; // OnChip
    }
}

void ProfilingHandler::GetProfTaskSpecificInfo(const TaskInfo &taskInfo, HCCLReportData &hcclReportData) const
{
    const auto &taskType = taskInfo.taskParam_.taskType;
    const auto &taskPara = taskInfo.taskParam_.taskPara;
    if (taskType == TaskParamType::TASK_SDMA || taskType == TaskParamType::TASK_RDMA) {
        hcclReportData.profInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src));
        hcclReportData.profInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst));
        hcclReportData.profInfo.dataSize = static_cast<u32>(taskPara.DMA.size);
        hcclReportData.profInfo.notifyID = taskPara.DMA.notifyID;
        hcclReportData.profInfo.linkType = static_cast<uint16_t>(taskPara.DMA.linkType);
    } else if (taskType == TaskParamType::TASK_REDUCE_INLINE || taskType == TaskParamType::TASK_REDUCE_TBE) {
        hcclReportData.profInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.src));
        hcclReportData.profInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.dst));
        hcclReportData.profInfo.dataSize = static_cast<u32>(taskPara.Reduce.size);
        hcclReportData.profInfo.notifyID = taskPara.Reduce.notifyID;
        hcclReportData.profInfo.linkType = static_cast<uint16_t>(taskPara.Reduce.linkType);
    } else if (taskType == TaskParamType::TASK_NOTIFY_RECORD || taskType == TaskParamType::TASK_NOTIFY_WAIT) {
        hcclReportData.profInfo.notifyID = taskPara.Notify.notifyID;
    } else if (taskType == TaskParamType::TASK_CCU) {
        HCCL_INFO("current taskType is TASK_CCU");
        ReportCcuInfo(taskInfo);
    }
}

constexpr u32 DPU_DEV_ID_BIT_POS = 12;    // 移位
void ProfilingHandler::GetDpuProfInfo(const TaskInfo &taskInfo, HCCLReportData &hcclReportData) const
{
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[ProfilingHandler::GetDpuProfInfo] taskInfo.dfxOpInfo_ is nullptr!");
        return;
    }
    const auto &taskType = taskInfo.taskParam_.taskType;
    const auto &taskPara = taskInfo.taskParam_.taskPara;
    hcclReportData.dpuProfInfo.itemId     = hcclReportData.profInfo.itemId;
    hcclReportData.dpuProfInfo.cclTag     = hcclReportData.profInfo.cclTag;
    hcclReportData.dpuProfInfo.groupName  = hcclReportData.profInfo.groupName;
    hcclReportData.dpuProfInfo.localRank  = hcclReportData.profInfo.localRank;
    hcclReportData.dpuProfInfo.remoteRank = hcclReportData.profInfo.remoteRank;
    hcclReportData.dpuProfInfo.rankSize   = hcclReportData.profInfo.rankSize;
    hcclReportData.dpuProfInfo.workFlowMode = hcclReportData.profInfo.workFlowMode;
    hcclReportData.dpuProfInfo.planeID    = hcclReportData.profInfo.planeID;
    hcclReportData.dpuProfInfo.stage      = hcclReportData.profInfo.stage;
    hcclReportData.dpuProfInfo.role       = hcclReportData.profInfo.role;
    hcclReportData.dpuProfInfo.durationEstimated = hcclReportData.profInfo.durationEstimated;
    hcclReportData.dpuProfInfo.dataType   = static_cast<uint32_t>(taskInfo.dfxOpInfo_->op_.dataType);
    hcclReportData.dpuProfInfo.opType     = static_cast<uint32_t>(taskInfo.dfxOpInfo_->op_.opType);
    hcclReportData.dpuProfInfo.transportType = static_cast<uint32_t>(SimpleTaskType::ROCE);
    hcclReportData.dpuProfInfo.aicpu_task_id = taskInfo.taskParam_.aicpuTaskId;
    hcclReportData.dpuProfInfo.npuDevId = taskInfo.taskParam_.npuDevId;
    hcclReportData.dpuProfInfo.dpuDevId = (1U << DPU_DEV_ID_BIT_POS);
    hcclReportData.dpuProfInfo.linkType = static_cast<uint16_t>(taskPara.DMA.linkType);
    if (hcclReportData.dpuProfInfo.remoteRank == INVALID_VALUE_RANKID) {
        hcclReportData.dpuProfInfo.transportType = static_cast<uint32_t>(SimpleTaskType::LOCAL);
        hcclReportData.dpuProfInfo.linkType = 0; // OnChip
    }
    hcclReportData.dpuProfInfo.taskId = taskInfo.taskId_;
    hcclReportData.dpuProfInfo.streamId = taskInfo.streamId_;
    hcclReportData.dpuProfInfo.timeStamp = taskInfo.taskParam_.beginTime;
    HCCL_INFO("[GetHCCLReportData]taskId[%llu], streamId[%lu], npuDevId[%lu], dpuDevId[%lu], "
              "starttime[%llu], endtime[%llu], aicputaskId[%llu].",
        hcclReportData.dpuProfInfo.taskId, hcclReportData.dpuProfInfo.streamId,
        hcclReportData.dpuProfInfo.npuDevId, hcclReportData.dpuProfInfo.dpuDevId,
        hcclReportData.dpuProfInfo.timeStamp, hcclReportData.ts, hcclReportData.dpuProfInfo.aicpu_task_id);
    if (taskType == TaskParamType::TASK_DPU_INLINE_WRITE
        || taskType == TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY) {
        hcclReportData.dpuProfInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src));
        hcclReportData.dpuProfInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst));
        hcclReportData.dpuProfInfo.dataSize = static_cast<u32>(taskPara.DMA.size);
        hcclReportData.dpuProfInfo.notifyID = taskPara.DMA.notifyID;
    } else if (taskType == TaskParamType::TASK_DPU_NOTIFY_WAIT
               || taskType == TaskParamType::TASK_DPU_CHANNEL_FENCE) {
        hcclReportData.dpuProfInfo.notifyID = taskPara.Notify.notifyID;
    }
}

void ProfilingHandler::GetHCCLReportData(const TaskInfo &taskInfo, HCCLReportData &hcclReportData) const
{
    HCCL_INFO("[ProfilingHandler]GetHCCLReportData start.");
    GetProfCommonInfo(taskInfo, hcclReportData);
    const auto &taskType = taskInfo.taskParam_.taskType;
    if (taskType == TaskParamType::TASK_DPU_INLINE_WRITE
        || taskType == TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY
        || taskType == TaskParamType::TASK_DPU_NOTIFY_WAIT
        || taskType == TaskParamType::TASK_DPU_CHANNEL_FENCE) {
        GetDpuProfInfo(taskInfo, hcclReportData);
    } else {
        GetProfTaskSpecificInfo(taskInfo, hcclReportData);
    }
    DumpHCCLReportData(taskInfo, hcclReportData);
    HCCL_INFO("[ProfilingHandler]GetHCCLReportData end.");
}

void ProfilingHandler::DumpHCCLReportData(const TaskInfo &taskInfo, const HCCLReportData &hcclReportData) const
{
    HCCL_INFO(
        "HCCLReportData profInfo: ts[%llu], itemId[%llu], cclTag[%llu], groupName[%llu], "
        "localRank[%u], remoteRank[%u], rankSize[%u], workFlowMode[%u], planeID[%u], ctxId[%u], "
        "stage[%u], role[%u], durationEstimated[%f], taskType[%d]",
        hcclReportData.ts, hcclReportData.profInfo.itemId, hcclReportData.profInfo.cclTag,
        hcclReportData.profInfo.groupName, hcclReportData.profInfo.localRank, hcclReportData.profInfo.remoteRank,
        hcclReportData.profInfo.rankSize, hcclReportData.profInfo.workFlowMode, hcclReportData.profInfo.planeID,
        hcclReportData.profInfo.ctxId, hcclReportData.profInfo.stage, hcclReportData.profInfo.role,
        hcclReportData.profInfo.durationEstimated, taskInfo.taskParam_.taskType);
    HCCL_INFO(
        "HCCLReportData profInfo detail: srcAddr[%llu], dstAddr[%llu], dataSize[%llu], notifyID[%llu], "
        "linkType[%u], opType[%s], transportType[%u], dataType[%s], rdmaType[%u]",
        hcclReportData.profInfo.srcAddr, hcclReportData.profInfo.dstAddr, hcclReportData.profInfo.dataSize,
        hcclReportData.profInfo.notifyID, hcclReportData.profInfo.linkType,
        OpTypeToSerialString(hcclReportData.profInfo.opType).c_str(), hcclReportData.profInfo.transportType,
        DataTypeToSerialString(hcclReportData.profInfo.dataType).c_str(), hcclReportData.profInfo.rdmaType);

    HCCL_INFO(
        "HCCLReportData dpuProfInfo: ts[%llu], itemId[%llu], cclTag[%llu], groupName[%llu], "
        "localRank[%u], remoteRank[%u], rankSize[%u], workFlowMode[%u], planeID[%u], "
        "stage[%u], role[%u], durationEstimated[%f], taskType[%d]",
        hcclReportData.ts, hcclReportData.dpuProfInfo.itemId, hcclReportData.dpuProfInfo.cclTag,
        hcclReportData.dpuProfInfo.groupName, hcclReportData.dpuProfInfo.localRank, hcclReportData.dpuProfInfo.remoteRank,
        hcclReportData.dpuProfInfo.rankSize, hcclReportData.dpuProfInfo.workFlowMode, hcclReportData.dpuProfInfo.planeID,
        hcclReportData.dpuProfInfo.stage, hcclReportData.dpuProfInfo.role, hcclReportData.dpuProfInfo.durationEstimated,
        taskInfo.taskParam_.taskType);
    HCCL_INFO(
        "HCCLReportData dpuProfInfo detail: srcAddr[%llu], dstAddr[%llu], dataSize[%llu], notifyID[%llu], "
        "linkType[%u], opType[%s], transportType[%u], dataType[%s], rdmaType[%u], "
        "taskId[%u], aicpu_task_id[%u], streamId[%u], npuDevId[%u], dpuDevId[%u], timeStamp[%llu]",
        hcclReportData.dpuProfInfo.srcAddr, hcclReportData.dpuProfInfo.dstAddr, hcclReportData.dpuProfInfo.dataSize,
        hcclReportData.dpuProfInfo.notifyID, hcclReportData.dpuProfInfo.linkType,
        OpTypeToSerialString(hcclReportData.dpuProfInfo.opType).c_str(), hcclReportData.dpuProfInfo.transportType,
        DataTypeToSerialString(hcclReportData.dpuProfInfo.dataType).c_str(), hcclReportData.dpuProfInfo.rdmaType,
        hcclReportData.dpuProfInfo.taskId, hcclReportData.dpuProfInfo.aicpu_task_id, hcclReportData.dpuProfInfo.streamId,
        hcclReportData.dpuProfInfo.npuDevId, hcclReportData.dpuProfInfo.dpuDevId, hcclReportData.dpuProfInfo.timeStamp);
}

void ProfilingHandler::ReportCcuInfo(const TaskInfo &taskInfo) const
{
    HCCL_INFO("[ProfilingHandler]ReportCcuInfo start.");
    if (taskInfo.taskParam_.ccuDetailInfo == nullptr) {
        HCCL_ERROR("[ProfilingHandler]ReportCcuInfo ccuDetailInfo is nullptr.");
        return;
    }
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[ProfilingHandler]ReportCcuInfo dfxOpInfo_ is nullptr.");
        return;
    }
    auto ccuDetailInfo = taskInfo.taskParam_.ccuDetailInfo;
    for (const auto &info : *ccuDetailInfo) {
        if (info.type == 0 && enableHcclL1_) {
            GetCcuTaskInfo(taskInfo, info);
        } else if (info.type == 1 &&  enableHcclL1_) {
            GetCcuWaitSignalInfo(taskInfo, info);
        } else if (info.type == CCU_TYPE && enableHcclL1_) {
            GetCcuGroupInfo(taskInfo, info);
        }
    }
    HCCL_INFO("[ProfilingHandler]ReportCcuInfo end.");
}

void ProfilingHandler::GetCcuTaskInfo(const TaskInfo &taskInfo, const CcuProfilingInfo &info) const
{
    HCCL_INFO("[ProfilingHandler]GetCcuTaskInfo start.");
    MsprofCcuTaskInfo ccuTaskInfo{};
    ccuTaskInfo.version       = 0;
    ccuTaskInfo.workFlowMode  = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    ccuTaskInfo.itemId        = GetProfHashId(info.name.c_str(), info.name.length());
    uint64_t groupName        = GetProfHashId(taskInfo.dfxOpInfo_->op_.opTag.c_str(),
                                              taskInfo.dfxOpInfo_->op_.opTag.length());
    ccuTaskInfo.groupName     = groupName;
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        ccuTaskInfo.rankId   = taskInfo.dfxOpInfo_->op_.myRank;
        ccuTaskInfo.ranksize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl *commImp = static_cast<CommunicatorImpl *>(taskInfo.dfxOpInfo_->comm_);
        ccuTaskInfo.rankId        = commImp->GetIdIndex(); 
        ccuTaskInfo.ranksize      = commImp->GetRankSize();
    }

    ccuTaskInfo.streamId      = taskInfo.streamId_;
    ccuTaskInfo.taskId        = taskInfo.taskId_;
    ccuTaskInfo.dieId         = info.dieId;
    ccuTaskInfo.missionId     = info.missionId;
    ccuTaskInfo.instrId       = info.instrId;
    // 上报数据
    uint64_t timestamp = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO("[ProfilingHandler]GetCcuTaskInfo, ccuTaskInfo data is: version[%u], workFlowMode[%u], itemId[%llu], "
              "groupName[%llu], rankId[%u], ranksize[%u], streamId[%u], taskId[%u], dieId[%u], "
              "missionId[%u],instrId[%u]",
              ccuTaskInfo.version, ccuTaskInfo.workFlowMode, ccuTaskInfo.itemId, ccuTaskInfo.groupName,
              ccuTaskInfo.rankId, ccuTaskInfo.ranksize, ccuTaskInfo.streamId, ccuTaskInfo.taskId, ccuTaskInfo.dieId,
              ccuTaskInfo.missionId, ccuTaskInfo.instrId);
    ReportAdditionInfo(MSPROF_REPORT_CCU_TASK_INFO, timestamp, &ccuTaskInfo, sizeof(ccuTaskInfo));
    HCCL_INFO("[ProfilingHandler]GetCcuTaskInfo end.");
}

void ProfilingHandler::GetCcuGroupInfo(const TaskInfo &taskInfo, const CcuProfilingInfo &info) const
{
    HCCL_INFO("[ProfilingHandler]GetCcuGroupInfo start.");
    MsprofCcuGroupInfo ccuGroupInfo{};
    ccuGroupInfo.version = 0;
    ccuGroupInfo.itemId  = GetProfHashId(info.name.c_str(), info.name.length());
    uint64_t groupName = GetProfHashId(taskInfo.dfxOpInfo_->op_.opTag.c_str(), taskInfo.dfxOpInfo_->op_.opTag.length());
    ccuGroupInfo.groupName      = groupName;
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        ccuGroupInfo.rankId   = taskInfo.dfxOpInfo_->op_.myRank;
        ccuGroupInfo.ranksize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl *commImp = static_cast<CommunicatorImpl *>(taskInfo.dfxOpInfo_->comm_);
        ccuGroupInfo.rankId        = commImp->GetIdIndex(); 
        ccuGroupInfo.ranksize      = commImp->GetRankSize();
    }
    ccuGroupInfo.workFlowMode   = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    ccuGroupInfo.streamId       = taskInfo.streamId_;
    ccuGroupInfo.taskId         = taskInfo.taskId_;
    ccuGroupInfo.dieId          = info.dieId;
    ccuGroupInfo.instrId        = info.instrId;
    ccuGroupInfo.missionId      = info.missionId;
    ccuGroupInfo.reduceOpType   = info.reduceOpType;
    ccuGroupInfo.inputDataType  = info.inputDataType;
    ccuGroupInfo.outputDataType = info.outputDataType;
    ccuGroupInfo.dataSize       = info.dataSize;
    std::copy(info.channelId, info.channelId + CCU_MAX_CHANNEL_NUM, ccuGroupInfo.channelId);
    std::copy(info.remoteRankId, info.remoteRankId + CCU_MAX_CHANNEL_NUM, ccuGroupInfo.remoteRankId);
    // 上报数据
    DumpCcuGroupInfo(ccuGroupInfo);
    uint64_t timestamp = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    ReportAdditionInfo(MSPROF_REPORT_CCU_GROUP_INFO, timestamp, &ccuGroupInfo, sizeof(ccuGroupInfo));
    HCCL_INFO("[ProfilingHandler]GetCcuGroupInfo end.");
}

void ProfilingHandler::DumpCcuGroupInfo(const MsprofCcuGroupInfo &ccuGroupInfo) const
{
    HCCL_INFO("[ProfilingHandler]GetCcuGroupInfo, ccuGroupInfo data is: version[%u], itemId[%llu], "
              "groupName[%llu], rankId[%u], ranksize[%u], workFlowMode[%u], streamId[%llu], taskId[%u], "
              "dieId[%u],instrId[%u],missionId[%u], dataSize[%llu]",
              ccuGroupInfo.version, ccuGroupInfo.itemId, ccuGroupInfo.groupName,
              ccuGroupInfo.rankId, ccuGroupInfo.ranksize, ccuGroupInfo.workFlowMode,
              ccuGroupInfo.streamId, ccuGroupInfo.taskId, ccuGroupInfo.dieId, ccuGroupInfo.instrId,
              ccuGroupInfo.missionId, ccuGroupInfo.dataSize);
    if (ccuGroupInfo.reduceOpType != INVALID_TYPE_VALUE) {
        HCCL_INFO("ccuGroupInfo reduceOpType is [%d]", static_cast<int>(ccuGroupInfo.reduceOpType));
    }
    if (ccuGroupInfo.inputDataType != INVALID_TYPE_VALUE) {
        HCCL_INFO("ccuGroupInfo inputDataType is [%d]", static_cast<int>(ccuGroupInfo.inputDataType));
    }
    if (ccuGroupInfo.outputDataType != INVALID_TYPE_VALUE) {
        HCCL_INFO("ccuGroupInfo outputDataType is [%d]", static_cast<int>(ccuGroupInfo.outputDataType));
    }
    for (auto i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
        if (ccuGroupInfo.channelId[i] != INVALID_VALUE_CHANNELID
            && ccuGroupInfo.remoteRankId[i] != INVALID_RANKID) {
            HCCL_INFO("[ProfilingHandler]GetCcuGroupInfo, ccuGroupInfo data is: channelId[%d] =  %u, "
                      "remoteRankId[%d] = %u",
                      i, ccuGroupInfo.channelId[i], i, ccuGroupInfo.remoteRankId[i]);
        }
    }
}

void ProfilingHandler::GetCcuWaitSignalInfo(const TaskInfo &taskInfo, const CcuProfilingInfo &info) const
{
    HCCL_INFO("[ProfilingHandler]GetCcuWaitSignalInfo start.");
    MsprofCcuWaitSignalInfo waitSignalInfo{};
    waitSignalInfo.version      = 0;
    waitSignalInfo.itemId       = GetProfHashId(info.name.c_str(), info.name.length());
    uint64_t groupName = GetProfHashId(taskInfo.dfxOpInfo_->op_.opTag.c_str(), taskInfo.dfxOpInfo_->op_.opTag.length());
    waitSignalInfo.groupName    = groupName;
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        waitSignalInfo.rankId   = taskInfo.dfxOpInfo_->op_.myRank;
        waitSignalInfo.ranksize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl *commImp = static_cast<CommunicatorImpl *>(taskInfo.dfxOpInfo_->comm_);
        waitSignalInfo.rankId        = commImp->GetIdIndex(); 
        waitSignalInfo.ranksize      = commImp->GetRankSize();
    }
    waitSignalInfo.workFlowMode = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    waitSignalInfo.streamId  = taskInfo.streamId_;
    waitSignalInfo.taskId    = taskInfo.taskId_;
    waitSignalInfo.dieId     = info.dieId;
    waitSignalInfo.instrId   = info.instrId;
    waitSignalInfo.missionId = info.missionId;
    waitSignalInfo.ckeId     = info.ckeId;
    waitSignalInfo.mask      = info.mask;
    std::copy(info.channelId, info.channelId + CCU_MAX_CHANNEL_NUM, waitSignalInfo.channelId);
    std::copy(info.remoteRankId, info.remoteRankId + CCU_MAX_CHANNEL_NUM, waitSignalInfo.remoteRankId);
    // 上报数据
    uint64_t timestamp = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO(
        "[ProfilingHandler]GetCcuWaitSignalInfo, waitSignalInfo data is: version[%u], itemId[%llu], groupName[%llu], "
        "rankId[%u], ranksize[%u], workFlowMode[%u], streamId[%llu], taskId[%u], dieId[%u],instrId[%u],missionId[%u], "
        "ckeId[%u],mask[%u]",
        waitSignalInfo.version, waitSignalInfo.itemId, waitSignalInfo.groupName, waitSignalInfo.rankId, waitSignalInfo.ranksize,
        waitSignalInfo.workFlowMode, waitSignalInfo.streamId, waitSignalInfo.taskId, waitSignalInfo.dieId, waitSignalInfo.instrId,
        waitSignalInfo.missionId, waitSignalInfo.ckeId, waitSignalInfo.mask);
    for (auto i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
        if (waitSignalInfo.channelId[i] != INVALID_VALUE_CHANNELID && waitSignalInfo.remoteRankId[i] != INVALID_RANKID) {
            HCCL_INFO(
                "[ProfilingHandler]GetCcuWaitSignalInfo, waitSignalInfo data is: channelId[%d] =  %u, remoteRankId[%d] = %u",
                i, waitSignalInfo.channelId[i], i, waitSignalInfo.remoteRankId[i]);
        }
    }
    ReportAdditionInfo(MSPROF_REPORT_CCU_WAIT_SIGNAL_INFO, timestamp, &waitSignalInfo, sizeof(waitSignalInfo));
    HCCL_INFO("[ProfilingHandler]GetCcuWaitSignalInfo end.");
}

void ProfilingHandler::ReportAclApi(uint32_t cmdType, uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId) const
{
    HCCL_INFO("[ProfilingHandler]ReportAclApi start.");
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_ACL_LEVEL;
    reporterData.type = static_cast<int32_t>(cmdType) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE;
    reporterData.threadId = threadId;
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    reporterData.itemId = cmdItemId;

    HCCL_INFO("[ProfilingHandler][ReportAclApi], reporterData data is: level[%u], type[%u], threadId[%u], beginTime "
              "[%llu], endTime[%llu], itemId[%llu]",
              reporterData.level, reporterData.type, reporterData.threadId, reporterData.beginTime,
              reporterData.endTime, reporterData.itemId);
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(1, &reporterData);
    HCCL_INFO("[ProfilingHandler][ReportAclApi], return value[%d]", ret);
    if (ret != 0) {
        THROW<InternalException>("Call dlMsprofReportApi failed, return[%d]", ret);
    }
    HCCL_INFO("[ProfilingHandler]ReportAclApi end.");
}

void ProfilingHandler::ReportNodeApi(uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId,
                                     bool cachedReq)
{
    HCCL_INFO("[ProfilingHandler]ReportNodeApi start.");
    // 获取数据
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    reporterData.threadId = threadId;
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    reporterData.itemId = cmdItemId;

    // 非单算子缓存数据
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cachedTaskApiInfoMutex_);
        cachedTaskApiInfo_.push(reporterData);
    } 
    if (!enableHostApi_) {
        return;
    }

    // 数据上报
    HCCL_INFO("[ProfilingHandler][ReportNodeApi], reporterData data is: level[%u], type[%u], threadId[%u],"
              "beginTime[%llu], endTime[%llu], itemId[%llu]",
              reporterData.level, reporterData.type, reporterData.threadId, reporterData.beginTime,
              reporterData.endTime, reporterData.itemId);
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(1, &reporterData);
    HCCL_INFO("[ProfilingHandler][ReportNodeApi], return value[%d]", ret);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportApi failed, return[%d]", ret);
    }
    HCCL_INFO("[ProfilingHandler]ReportNodeApi end.");
}

void ProfilingHandler::ReportNodeBasicInfo(uint64_t timeStamp, uint64_t cmdItemId, uint32_t threadId, bool cachedReq)
{
    // 获取数据
    MsprofCompactInfo reporterData{};
    reporterData.level     = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type      = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    reporterData.threadId  = threadId;
    reporterData.dataLen   = sizeof(MsprofNodeBasicInfo);
    reporterData.timeStamp = timeStamp;
    reporterData.data.nodeBasicInfo.opName   = cmdItemId;
    reporterData.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_HCCL;
    reporterData.data.nodeBasicInfo.opType   = cmdItemId;
    reporterData.data.nodeBasicInfo.opFlag   = 0;
    HCCL_INFO("[ProfilingHandler][ReportNodeBasicInfo], reporterData data is: level[%u], type[%u], threadId[%u], "
              "dataLen[%u], taskType[%u], opFlag[%u]", reporterData.level, reporterData.type, reporterData.threadId,
              reporterData.dataLen, reporterData.data.nodeBasicInfo.taskType, reporterData.data.nodeBasicInfo.opFlag);
    // 缓存数据
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cacheHcclOpInfoMutex_);
        cacheHcclOpInfo_.push(reporterData);
    }
    if (!enableHcclL1_) {
        return;
    }
    // 数据上报
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportCompactInfo(1, &reporterData, sizeof(MsprofCompactInfo));
    HCCL_INFO("Call MsprofReportCompactInfo, return value[%d]", ret);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportCompactInfo failed, return[%d]", ret);
    }
    HCCL_INFO("[ProfilingHandler]ReportNodeBasicInfo end.");
}

void ProfilingHandler::ReportHcclOpInfo(uint64_t timeStamp, const DfxOpInfo &opInfo, uint32_t threadId, bool cachedReq)
{
    // 获取数据
    MsprofCompactInfo reporterData{};
    reporterData.level     = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type      = MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE;
    reporterData.threadId  = threadId;
    reporterData.dataLen   = sizeof(MsprofHCCLOPInfo);
    reporterData.timeStamp = timeStamp;
    reporterData.data.hcclopInfo.relay    = 0;
    reporterData.data.hcclopInfo.retry    = 0;
    reporterData.data.hcclopInfo.dataType = opInfo.op_.dataType;
    reporterData.data.hcclopInfo.algType  = GetProfHashId(opInfo.algType_.c_str(), opInfo.algType_.length());
    uint64_t groupName                     = GetProfHashId(opInfo.op_.opTag.c_str(), opInfo.op_.opTag.length());
    reporterData.data.hcclopInfo.groupName = groupName;
    u32 ranksize{0};
    if (opInfo.isIndop_ == true) {
        ranksize = opInfo.rankSize_;
        reporterData.data.hcclopInfo.count = opInfo.op_.dataCount;
    } else {
        CommunicatorImpl *commImp = static_cast<CommunicatorImpl *>(opInfo.comm_);
        ranksize = commImp->GetRankSize();
        if (opInfo.op_.opType == OpType::ALLTOALLV) {
            u64 sendCount = 0;
            for (u64 i = 0; i < ranksize; i++) {
                sendCount += *(static_cast<const u64 *>(opInfo.op_.all2AllVDataDes.sendCounts) + i);
            }
            reporterData.data.hcclopInfo.count = sendCount;
        } else if (opInfo.op_.opType == OpType::ALLTOALL) {
            reporterData.data.hcclopInfo.count = opInfo.op_.all2AllDataDes.sendCount;
        } else {
            reporterData.data.hcclopInfo.count = opInfo.op_.dataCount;
        }
    }
    // 缓存数据
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cacheHcclOpInfoMutex_);
        cacheHcclOpInfo_.push(reporterData);
    }
    if (!enableHostApi_) {
        return;
    }
    // 数据上报
    HCCL_INFO("[ProfilingHandler][ReportHcclOpInfo], data is: level[%u], type[%u], threadId[%u], dataLen[%u], "
              "timeStamp[%llu], relay [%u], retry[%u], dataType[%s], algType[%u], groupName[%llu], count[%llu]",
              reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen,
              reporterData.timeStamp, reporterData.data.hcclopInfo.relay, reporterData.data.hcclopInfo.retry,
              DataTypeToSerialString(reporterData.data.hcclopInfo.dataType).c_str(), reporterData.data.hcclopInfo.algType,
              reporterData.data.hcclopInfo.groupName, reporterData.data.hcclopInfo.count);
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportCompactInfo(1, &reporterData, sizeof(MsprofCompactInfo));
    if (ret != 0) {
         THROW<InternalException>("[ProfilingHandler] Call dlMsprofReportCompactInfo failed, return[%d]", ret);
    }
}

void ProfilingHandler::ReportAdditionInfo(uint32_t type, uint64_t timeStamp, void *data, uint32_t len) const
{
    HCCL_INFO("[ProfilingHandler]ReportAdditionInfo start.");
    MsprofAdditionalInfo reporterData{};
    reporterData.level     = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type      = type;
    reporterData.threadId  = SalGetTid();
    reporterData.dataLen   = len;
    reporterData.timeStamp = timeStamp;
    s32 sret               = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    if (sret != EOK) {
        THROW<InternalException>("Call memcpy_s failed, errorno[%d]", sret);
    }
    HCCL_INFO(
        "[ProfilingHandler][ReportAdditionInfo], level [%u], type[%u], threadId[%u], dataLen[%u], timeStamp[%llu]",
        reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen, reporterData.timeStamp);
    s32 ret
        = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(0, &reporterData, sizeof(MsprofAdditionalInfo));
    HCCL_INFO("Call MsprofReportAdditionalInfo, return value[%d]", ret);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportAdditionalInfo failed, return[%d]", ret);
    }
    HCCL_INFO("[ProfilingHandler]ReportAdditionInfo end.");
}

int32_t ProfilingHandler::CommandHandle(uint32_t rtType, void *data, uint32_t len) const
{
    (void)len;
    if (data == nullptr || rtType != rtProfCtrlType_t::RT_PROF_CTRL_SWITCH) {
        HCCL_ERROR("[ProfilingHandler][CommandHandle] data is nullptr or rtType is invalid, rtType[%u]", rtType);
        return HCCL_E_PARA;
    }
    rtProfCommandHandle_t *profConfigParam = reinterpret_cast<rtProfCommandHandle_t *>(data);
    auto type = profConfigParam->type;
    auto profconfig = profConfigParam->profSwitch;
    HCCL_RUN_INFO("[Profiling][CommandHandle] CommandHandle's rtType is %u. CommandHandle_switch type[%u], " \
            "profconfig[%u], deviceLogicId[%u]", rtType, type, profconfig, profConfigParam->devIdList[0]);
    switch (type) {
        case PROF_COMMANDHANDLE_TYPE_START:
            instance_.StartSubscribe(profconfig);
            break;
        case PROF_COMMANDHANDLE_TYPE_STOP:
            instance_.StopSubscribe(); 
            break;
        default:
            HCCL_RUN_INFO("[Profiling][CommandHandle] Unexpected behaviour.");
    }
    return HCCL_SUCCESS;
}

void ProfilingHandler::StartSubscribe(uint64_t profconfig)
{
    // enableHostApi_打开时，HostApi粒度的打点控制
    HCCL_RUN_INFO("[Profiling][CommandHandle] profSwitch is[%llu]", profconfig);
    if ((profconfig & PROF_ACL_API_MASK) != 0) {
        StartHostApiSubscribe();
    }
    // 集合通信算子粒度的打点 只有L0打开的时候才上报 L1打开的时候不上报; AICPU也不上报算子粒度的打点
    if ((profconfig & PROF_TASK_TIME_MASK) != 0 && (profconfig & PROF_TASK_TIME_L1_MASK) == 0) {
        StartHostHcclOpSubscribe();
    }
    // L1打开时, 上报task粒度的打点和子task的详细信息
    if ((profconfig & PROF_TASK_TIME_L1_MASK) != 0) {
        StartTaskApiSubscribe();
        StartAdditionInfoSubscribe();
        StartCcuSubscribe(); // ccu开关等级改为L1
    } 
    HCCL_RUN_INFO("[Profiling][CommandHandle] profSwitch is[%llu]", profconfig);
}

void ProfilingHandler::StartHostApiSubscribe()
{
    enableHostApi_ = true;
    CallProfRegHostApi();
    ReportStoragedCompactInfo(); // 缓存信息上报
    ReportMc2AdditionInfo();
    HCCL_RUN_INFO("SetHostApiSubscribe:[%d]", enableHostApi_);
}

void ProfilingHandler::CallProfRegHostApi() const
{
    if (!enableHostApi_) {
        return;
    }
    auto &profFunction = DlProfFunction::GetInstance();
    for (auto i = 0; i < OpType::__COUNT__; ++i) {
        OpType type(static_cast<OpType::Value>(i));
        s32           ret = profFunction.dlMsprofRegTypeInfo(MSPROF_REPORT_ACL_LEVEL,
                                                             static_cast<uint32_t>(type) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE,
                                                             type.Describe().c_str());
        if (ret != 0) {
            THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", ret);
        }
    }
    for (auto i = 0; i < OpType::__COUNT__; ++i) {
        OpType type(static_cast<OpType::Value>(i));
        s32           ret = profFunction.dlMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL,
                                                             static_cast<uint32_t>(type) + MSPROF_REPORT_NODE_HCCL_BASE_TYPE,
                                                             type.Describe().c_str());
        if (ret != 0) {
            THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", ret);
        }
    }
    const std::string hcclType("hccl_op_info");
    s32 ret = profFunction.dlMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE,
                                               hcclType.c_str());
    if (ret != 0) {
        THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", ret);
    }
}

void ProfilingHandler::ReportStoragedCompactInfo()
{
    std::lock_guard<std::mutex> lock(cacheHcclOpInfoMutex_);
    HCCL_INFO("[ReportStoragedCompactInfo] The size of the storageCompactInfo_ is [%u]", cacheHcclOpInfo_.size());
    std::queue<MsprofCompactInfo> tempCompactInfo = cacheHcclOpInfo_;
    while (!tempCompactInfo.empty()) {
        MsprofCompactInfo reportData = tempCompactInfo.front();
        tempCompactInfo.pop();
        s32 ret = DlProfFunction::GetInstance().dlMsprofReportCompactInfo(0, &reportData, sizeof(MsprofCompactInfo));
        if (ret != 0) {
            THROW<InternalException>("Call MsprofRegTypeInfo failed, return[%d]", ret);
        }
    }
}

void ProfilingHandler::ReportMc2AdditionInfo()
{
    std::lock_guard<std::mutex> lock(cacheHcclAdditionInfoMutex_);
    HCCL_INFO("[ReportMc2AdditionInfo] The size of the storageCompactInfo_ is [%u]", cacheHcclAdditionInfo_.size());
    std::queue<MsprofAdditionalInfo> tempCompactInfo = cacheHcclAdditionInfo_;
    while (!tempCompactInfo.empty()) {
        MsprofAdditionalInfo reportData = tempCompactInfo.front();
        tempCompactInfo.pop();
        s32 ret = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(1, &reportData,
                                                                             sizeof(MsprofAdditionalInfo));
        if (ret != 0) {
            THROW<InternalException>("Call MsprofRegTypeInfo failed, return[%d]", ret);
        }
    }
}

void ProfilingHandler::StartTaskApiSubscribe()
{
    enableHcclNode_ = true;
    CallProfRegTaskTypeApi();
    ReportStoragedTaskApi();
    HCCL_INFO("SetTaskApiSubscribe:[%d]", enableHcclNode_);
}

void ProfilingHandler::CallProfRegTaskTypeApi() const
{
    if (!enableHcclNode_) {
        HCCL_INFO("[ProfilingHandler] enableHostApi_ is false.");
        return;
    }
    const std::string hcclType("hccl_info");
    s32               sret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
        MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO), hcclType.c_str());
    if (sret != 0) {
        THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", sret);
    }

    const std::string dpuhcclType("dpu_hccl_info");
    sret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
        MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(ProfTaskType::TASK_DPU_HCCL_INFO), dpuhcclType.c_str());
    if (sret != 0) {
        THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", sret);
    }
    const std::vector<std::pair<uint32_t, std::string>> taskTypes
        = {{MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE, "context_id_info"}};
    const std::vector<std::pair<uint32_t, std::string>> taskOtherTypes
        = {{MSPROF_REPORT_NODE_BASIC_INFO_TYPE, "node_basic_info"},
           {MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE, "mc2_comm_info"}};

    for (auto &it : taskTypes) {
        s32 ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(MSPROF_REPORT_HCCL_NODE_LEVEL, it.first,
                                                                    it.second.c_str());
        if (ret != 0) {
            THROW<InternalException>("Call dlMsprofRegTypeInfo failed, return[%d]", ret);
        }
    }
    for (auto &it : taskOtherTypes) {
        s32 ret
            = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, it.first, it.second.c_str());
        if (ret != 0) {
            THROW<InternalException>("Call dlMsprofRegTypeInfo failed, return[%d]", ret);
        }
    }
}

void ProfilingHandler::ReportStoragedTaskApi()
{
    std::lock_guard<std::mutex> lock(cachedTaskApiInfoMutex_);
    HCCL_INFO("[ReportStoragedTaskApi] taskApiQueueSize is [%u]", cachedTaskApiInfo_.size());
    if (!cachedTaskApiInfo_.empty()) {
        std::queue<MsprofApi> tempTaskApi = cachedTaskApiInfo_;
        while (!tempTaskApi.empty()) {
            MsprofApi reportData = tempTaskApi.front();
            tempTaskApi.pop();
            s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(0, &reportData);
            if (ret != 0) {
                THROW<InternalException>("Call dlMsprofReportApi failed, return[%d]", ret);
            }
        }
    }
}

void ProfilingHandler::StartHostHcclOpSubscribe() {
    enableHcclNode_ = true; // Node_ = L0 | L1
    enableHcclL0_ = true;
    CallProfRegHcclOpApi();
    ReportStoragedCompactInfo(); // 缓存信息上报
    HCCL_RUN_INFO("StartHostHcclOpSubscribe:[%d]", enableHcclNode_);
}

void ProfilingHandler::CallProfRegHcclOpApi() const
{
    if (enableHcclL0_ == false) {
        HCCL_INFO("[ProfilingHandler] enableHcclNode_ is false.");
        return;
    }
    for (auto i = 0; i < OpType::__COUNT__; ++i) {
        OpType type(static_cast<OpType::Value>(i));
        s32 ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
            MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(type) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE,
            type.Describe().c_str());
        if (ret != 0) {
            THROW<InternalException>("[ProfilingHandler]Call MsprofReportApi fail, return[%d]", ret);
        }
    }
    s32 ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL,
                                                                MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE, "mc2_comm_info");
    if (ret != 0) {
        THROW<InternalException>("[ProfilingHandler]Call MsprofRegTypeInfo fail, return[%d]", ret);
    }
}

void ProfilingHandler::StartAdditionInfoSubscribe()
{
    enableHcclL1_ = true;
    ReportStoragedAdditionInfo();
    HCCL_RUN_INFO("StartAdditionInfoSubscribe:[%d]", enableHcclL1_);
}

void ProfilingHandler::ReportStoragedAdditionInfo()
{
    std::lock_guard<std::mutex> lock(cacheTaskInfosMutex_);
    if (cacheTaskInfos_.empty()) {
        HCCL_INFO("[ProfilingHandler]ReportStoragedAdditionInfo cacheTaskInfos_ is empty.");
        return;
    }
    for (auto &taskInfo : cacheTaskInfos_) {
        HCCLReportData hcclReportData{};
        GetHCCLReportData(taskInfo, hcclReportData);
        // 调用additionInfo接口上报数据
        CallAdditionInfo(hcclReportData, &hcclReportData.profInfo, sizeof(hcclReportData.profInfo), ProfTaskType::TASK_HCCL_INFO);
    }
}

void ProfilingHandler::StartCcuSubscribe()
{
    enableHcclNode_ = true;
    enableHcclL1_ = true;
    HCCL_INFO("ProfilingHandler StartCcuSubscribe");
    const std::vector<std::pair<uint32_t, std::string>> ccuInfoTypes
        = {{MSPROF_REPORT_CCU_TASK_INFO, "ccu_task_info"},
           {MSPROF_REPORT_CCU_WAIT_SIGNAL_INFO, "ccu_wait_signal_info"},
           {MSPROF_REPORT_CCU_GROUP_INFO, "ccu_group_info"}};
    for (auto &it : ccuInfoTypes) {
        s32 ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(MSPROF_REPORT_HCCL_NODE_LEVEL, it.first,
                                                                    it.second.c_str());
        if (ret != 0) {
            THROW<InternalException>("Call dlMsprofRegTypeInfo failed, return[%d]", ret);
        }
    }
    std::lock_guard<std::mutex> lock(cacheTaskInfosMutex_);
    if (cacheTaskInfos_.empty()) {
        HCCL_INFO("[ProfilingHandler]StartL2Subscribe cacheTaskInfos_ is empty.");
        return;
    }
    for (auto &taskInfo : cacheTaskInfos_) {
        ReportCcuInfo(taskInfo);
    }
}

void ProfilingHandler::ProfilingHandler::StopSubscribe()
{
    enableHostApi_  = false;
    enableHcclNode_ = false;
    enableHcclL0_   = false;
    enableHcclL1_   = false;
    HCCL_RUN_INFO("[ProfilingHandler]StopSubscribe.");
}

bool ProfilingHandler::GetHostApiState() const
{
    return enableHostApi_;
}
bool ProfilingHandler::GetHcclNodeState() const
{
    return enableHcclNode_;
}
bool ProfilingHandler::GetHcclL0State() const
{
    return enableHcclL0_;
}

bool ProfilingHandler::GetHcclL1State() const
{
    return enableHcclL1_;
}

uint64_t ProfilingHandler::GetProfHashId(const char *name, uint32_t len) const
{
    if (name == nullptr || len == 0) {
        HCCL_WARNING("HashData is empty.  name:%s, len:%u", name, len);
        return INVALID_U64;
    }
    if (DlProfFunction::GetInstance().dlMsprofStr2Id == nullptr) {
        return INVALID_U64;
    }
    return DlProfFunction::GetInstance().dlMsprofStr2Id(name, len);
}

void ProfilingHandler::ReportHcclMC2CommInfo(const Stream &kfcStream, Stream &stream, 
                                             const std::vector<Stream *> &aicpuStreams, const std::string &id, 
                                             RankId myRank, u32 rankSize, RankId rankInParentComm)
{
    ProfilingDeviceCommResInfo hcclMc2Info;
    hcclMc2Info.groupName                    = GetProfHashId(id.c_str(), id.length());
    hcclMc2Info.rankSize                     = rankSize;
    hcclMc2Info.rankId                       = myRank;
    hcclMc2Info.usrRankId                    = rankInParentComm;
    hcclMc2Info.aicpuKfcStreamId             = static_cast<uint32_t>(kfcStream.GetSqId());
    hcclMc2Info.reserve                      = 0;
    const uint32_t ONCE_REPORT_STREAM_NUM_MAX = 8;
    for (uint32_t streamIndex = 0, reportId = 0; streamIndex < aicpuStreams.size(); streamIndex++) {
        HCCL_INFO("streamIndex:%u, reportId:%u, streamId(sqId):%u", streamIndex, reportId, aicpuStreams[streamIndex]->GetSqId());
        hcclMc2Info.commStreamIds[reportId++] = aicpuStreams[streamIndex]->GetSqId();
        if (reportId == ONCE_REPORT_STREAM_NUM_MAX) {
            hcclMc2Info.commStreamSize = reportId;
            ReportMc2AdditionInfo(DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info));
            reportId = 0;
        }
        if (streamIndex == (aicpuStreams.size() - 1)) {
            hcclMc2Info.commStreamIds[reportId++] = stream.GetSqId();
            hcclMc2Info.commStreamSize            = reportId;
            ReportMc2AdditionInfo(DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info,
                                 sizeof(hcclMc2Info));
            reportId = 0;
        }
    }
    if (aicpuStreams.empty()) {
        HCCL_INFO("only exist main stream, streamId(sqId):%u", stream.GetSqId());
        hcclMc2Info.commStreamIds[0] = stream.GetSqId();
        hcclMc2Info.commStreamSize   = 1; // 只有主流1条
        ReportMc2AdditionInfo(DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info));
    }
}

void ProfilingHandler::ReportHcclMC2CommInfo(const u32 kfcStreamId,
                            const std::vector<u32> &aicpuStreamsId, const std::string &id,
                            RankId myRank, u32 rankSize, RankId rankInParentComm)
{
    ProfilingDeviceCommResInfo hcclMc2Info;
    hcclMc2Info.groupName = GetProfHashId(id.c_str(),id.length());
    hcclMc2Info.rankSize = rankSize;
    hcclMc2Info.rankId = myRank;
    hcclMc2Info.usrRankId = rankInParentComm;
    hcclMc2Info.aicpuKfcStreamId = static_cast<uint32_t>(kfcStreamId);
    hcclMc2Info.reserve = 0;
    
    const uint32_t ONCE_REPORT_STREAM_NUM_MAX = 8;
    uint32_t reportId = 0;
    for (uint32_t streamIndex = 0; streamIndex < aicpuStreamsId.size(); streamIndex++) {
        HCCL_INFO("streamIndex:[%u], reportId:[%d], streamId:[%u] id [%s] hcclMC2Info.groupName:[%lu]", streamIndex, 
            reportId, aicpuStreamsId[streamIndex], id.c_str(), hcclMc2Info.groupName);
        hcclMc2Info.commStreamIds[reportId++] = aicpuStreamsId[streamIndex];
        if (reportId == ONCE_REPORT_STREAM_NUM_MAX) {
            hcclMc2Info.commStreamSize = reportId;
            ReportMc2AdditionInfo(DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info));
            reportId = 0;
        }
    }
    if (reportId > 0) {
        hcclMc2Info.commStreamSize = reportId;
        ReportMc2AdditionInfo(DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info,
        sizeof(hcclMc2Info));
        reportId = 0;
    }
}
void ProfilingHandler::ReportMc2AdditionInfo(uint64_t timeStamp, const void *data, int len)
{
    MsprofAdditionalInfo reporterData{};
    reporterData.level     = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type      = MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE;
    reporterData.threadId  = SalGetTid();
    reporterData.dataLen   = len;
    reporterData.timeStamp = timeStamp;
    s32 sret               = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    if (sret != EOK) {
        THROW<InternalException>("Call memcpy_s failed, errorno[%d]", sret);
    }
    HCCL_INFO("[ProfilingHandler][ReportMc2CommInfo], level [%u], type[%u], threadId[%u], dataLen[%u], timeStamp[%llu]",
              reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen,
              reporterData.timeStamp);
    if (!enableHostApi_) {  // 比较特殊，没开就存 跟A3保持一致
        std::lock_guard<std::mutex> lock(cacheHcclAdditionInfoMutex_);
        // 缓存对应数据
        cacheHcclAdditionInfo_.push(reporterData);
        return;
    }
    s32 ret
        = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(1, &reporterData, sizeof(MsprofAdditionalInfo));
    HCCL_INFO("Call MsprofReportAdditionalInfo, return value[%d]", ret);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportAdditionalInfo failed, return[%d]", ret);
    }
    HCCL_INFO("[ProfilingHandler]ReportMc2CommInfo end.");
}
 
} // namespace Hccl