/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hcclCommTaskExceptionLite.h"
#include "stream_lite.h"
#include "global_mirror_tasks.h"
#include "hcomm_task_scheduler_error.h"
#include "task_struct_v2.h"
#include "dlhal_function_v2.h"
#include "read_write_lock.h"
#include "aicpu_indop_env.h"

namespace hcomm {
constexpr u32 RT_SDMA_COMPERR = 0x9; // A3 sdma error类型为0x9时，表示写拷贝发生超时代答，或者数据搬移时地址译码错误
constexpr u32 RT_SDMA_COMPDATAERR = 0xa; // A3 sdma error类型为0xa时，表示读拷贝发生超时代答，或者读HBM返回ERROR
constexpr u32 RT_SDMA_DATAERR = 0x8; // A3 sdma error类型为0x8时，表示读HBM返回ERROR
constexpr u32 RT_UB_LOCAL_OPERATIOINERR = 0x2; // A5 ub error类型为0x2时，表示UB本端返回ERROR
constexpr u32 RT_UB_REMOTE_OPERATIOINERR = 0x3; // A5 ub error类型为0x3时，表示UB远端返回ERROR
constexpr u32 RT_UB_LINK_FAILEDERR = 0x5; // A5 ub error类型为0x5时，表示网络异常，taack超时
constexpr uint8_t ubSqeType = 9; // A5 sqeType为9表示UBDMA任务
constexpr uint8_t sdmaSqeType = 11; // A5 sqeType为11表示SDMA任务

constexpr uint32_t TASK_CONTEXT_SIZE = 50; // task 执行失败时打印前序task信息的数量
constexpr uint32_t TASK_CONTEXT_INFO_SIZE = LOG_TMPBUF_SIZE - TASK_CONTEXT_SIZE; // task 执行失败时打印前序task信息的长度限制
constexpr u32 MAX_NAME_LEN = 64;

HcclCommTaskExceptionLite &HcclCommTaskExceptionLite::GetInstance()
{
    static HcclCommTaskExceptionLite instance; // aicpu侧一个dev一个进程，不需要按dev区分单例对象
    return instance;
}

void HcclCommTaskExceptionLite::Init(u32 devId)
{
    devId_ = devId;
    HCCL_INFO("[%s]success, devId_[%u]", __func__, devId_);
}

void HcclCommTaskExceptionLite::Call()
{
    if (stopCall_ == true) {
        return;
    }

    HcclResult ret = HandleExceptionCqe();
    if (ret != HCCL_SUCCESS) {
        stopCall_ = true;
        HCCL_ERROR("[%s]HandleExceptionCqe fail, set stopCall_[%d]", __func__, stopCall_); // 函数调用失败，停止调用避免刷屏
    }
}

HcclResult HcclCommTaskExceptionLite::HandleExceptionCqe()
{
    ReadWriteLockBase &commAicpuMapMutex = AicpuIndopProcess::AicpuGetCommMutex();
    ReadWriteLock rwlock(commAicpuMapMutex);
    rwlock.readLock();

    std::vector<std::pair<std::string, CollCommAicpuMgr *>> aicpuCommInfo;
    CHK_RET(AicpuIndopProcess::AicpuGetCommAll(aicpuCommInfo));

    for (auto &commInfo : aicpuCommInfo) {
        CollCommAicpu *aicpuComm = commInfo.second->GetCollCommAicpu();
        CHK_PTR_NULL(aicpuComm);

        if (aicpuComm->GetCommmStatus() == HcclCommStatus::HCCL_COMM_STATUS_INVALID) {
            continue;
        }

        ReadWriteLockBase &commThreadMutex = aicpuComm->GetThreadMutex();
        ReadWriteLock threadRwlock(commThreadMutex);
        threadRwlock.readLock();
        const std::vector<std::shared_ptr<hccl::Thread>> threads = aicpuComm->GetAllThread();
        for (auto thread : threads) {
            rtLogicCqReport_t cqeException;
            dfx::CqeStatus cqeStatus = dfx::CqeStatus::kDefault;
            Hccl::StreamLite *streamLite = static_cast<Hccl::StreamLite *>(thread->GetStreamLitePtr());
            CHK_PTR_NULL(streamLite);

            HcclResult ret = GetThreadCqe(thread.get(), cqeException, cqeStatus);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]GetThreadCqe fail, aicpuComm[%s], streamId[%u]",
                __func__, aicpuComm->GetIdentifier().c_str(), streamLite->GetId()), ret);

            ret = ProcessCqe(aicpuComm, cqeException, cqeStatus, aicpuCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]ProcessCqe fail, aicpuComm[%s], streamId[%u], "
                "cqeStatus[%d]", __func__, aicpuComm->GetIdentifier().c_str(), streamLite->GetId(), cqeStatus), ret);
        }
        threadRwlock.readUnlock();
    }
    rwlock.readUnlock();
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::PrintAllCommTaskException()
{
    ReadWriteLockBase &commAicpuMapMutex = AicpuIndopProcess::AicpuGetCommMutex();
    ReadWriteLock rwlock(commAicpuMapMutex);
    rwlock.readLock();

    std::vector<std::pair<std::string, CollCommAicpuMgr *>> aicpuCommInfo;
    CHK_RET(AicpuIndopProcess::AicpuGetCommAll(aicpuCommInfo));

    HCCL_RUN_INFO("[TaskException][AICPU]%s start, comm size[%u]", __func__, aicpuCommInfo.size());
    HcclResult ret = HCCL_SUCCESS;
    for (auto &commInfo : aicpuCommInfo) {
        CollCommAicpu *aicpuComm = commInfo.second->GetCollCommAicpu();
        HcclResult pRet = PrintCommTaskException(aicpuComm);
        CHK_PRT_CONT(pRet != HCCL_SUCCESS, HCCL_ERROR("PrintCommTaskException fail, comm[%s]",
            aicpuComm->GetIdentifier().c_str()));
        ret = (pRet != HCCL_SUCCESS) ? pRet : ret;
    }
    HCCL_RUN_INFO("[TaskException][AICPU]%s end, ret[%d]", __func__, ret);
    rwlock.readUnlock();
    return ret;
}

HcclResult HcclCommTaskExceptionLite::PrintCommTaskException(CollCommAicpu *aicpuComm)
{
    CHK_PTR_NULL(aicpuComm);
    HcclResult ret = HCCL_SUCCESS;
    HCCL_RUN_INFO("[TaskException][AICPU]%s comm[%s] start", __func__, aicpuComm->GetIdentifier().c_str());
    ReadWriteLockBase &commThreadMutex = aicpuComm->GetThreadMutex();
    ReadWriteLock threadRwlock(commThreadMutex);
    threadRwlock.readLock();
    const std::vector<std::shared_ptr<hccl::Thread>> threads = aicpuComm->GetAllThread();
    for (auto thread : threads) {
        CHK_SMART_PTR_NULL(thread);
        Hccl::StreamLite *streamLite = static_cast<Hccl::StreamLite *>(thread->GetStreamLitePtr());
        CHK_PTR_NULL(streamLite);
        u32 sqHead = 0U;
        u32 sqTail = 0U;
        HcclResult ret = QuerySqStatus(devId_, streamLite->GetSqId(), sqHead, sqTail);
        if (ret != HCCL_SUCCESS || sqHead == sqTail) { // 此流为空时，不打印
            HCCL_RUN_INFO("[TaskException][AICPU]PrintTaskExceptionBySqeId skip, "
                "QuerySqStatus ret[%d], aicpuComm[%s], sqId[%u], sqHead[%u], sqTail[%u]",
                ret, aicpuComm->GetIdentifier().c_str(), streamLite->GetSqId(), sqHead, sqTail);
            continue;
        }
        uint16_t streamId = 0;
        uint16_t taskId = 0;
        streamLite->GetRtsq()->GetStreamIdAndTaskIdBySqIdx(sqHead, streamId, taskId);
        const u32 sqeId = GetSqeId(taskId, streamId);
        HcclResult pRet = PrintTaskExceptionBySqeId(aicpuComm, streamLite->GetSqId(), sqeId);
        CHK_PRT_CONT(pRet != HCCL_SUCCESS, HCCL_ERROR("PrintTaskExceptionBySqeId fail, comm[%s], sqId[%u], sqeId[%u]",
            aicpuComm->GetIdentifier().c_str(), streamLite->GetSqId(), sqeId));
        ret = (pRet != HCCL_SUCCESS) ? pRet : ret;
    }
    threadRwlock.readUnlock();
    HCCL_RUN_INFO("[TaskException][AICPU]%s comm[%s] end, ret[%d]", __func__, aicpuComm->GetIdentifier().c_str(), ret);
    return ret;
}

HcclResult HcclCommTaskExceptionLite::GetThreadCqe(hccl::Thread* thread, rtLogicCqReport_t &cqeException,
    dfx::CqeStatus &cqeStatus)
{
    CHK_SMART_PTR_NULL(thread);
    Hccl::StreamLite *streamLite = static_cast<Hccl::StreamLite *>(thread->GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);

    constexpr u32 reportSize = MAX_REPORT_CNT;
    rtLogicCqReport_t streamReport[reportSize];
    
    CqeQueryInput cqeQueryInput;
    cqeQueryInput.devId = devId_;
    cqeQueryInput.streamId = streamLite->GetId();
    cqeQueryInput.sqId = streamLite->GetSqId();
    cqeQueryInput.cqId = streamLite->GetCqId();
    cqeQueryInput.type = static_cast<uint32_t>(DRV_LOGIC_TYPE);
    cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t *>(streamReport);
    
    cqeStatus = CqReportRecv(cqeQueryInput, cqeException);
    if (cqeStatus == dfx::CqeStatus::kCqeInnerError) {
        HCCL_ERROR("[%s]CqReportRecv fail, CqeQueryInput:%s", __func__, cqeQueryInput.ToString().c_str());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::ProcessCqe(CollCommAicpu *aicpuComm, const rtLogicCqReport_t &exceptionInfo,
    const CqeStatus &cqeStatus, const std::vector<std::pair<std::string, CollCommAicpuMgr *>> &aicpuCommInfo)
{
    if (cqeStatus == dfx::CqeStatus::kDefault) {
        return HCCL_SUCCESS;
    }

    if (hcomm::GetTaskExceptionEnable() == false) {
        HCCL_ERROR("[TaskException][AICPU]taskException enable is false, skip print taskException");
        return HCCL_SUCCESS;
    }

    HcclResult ret = HCCL_SUCCESS;
    const u32 sqeId = GetSqeId(exceptionInfo.taskId, exceptionInfo.streamId);
    ret = PrintTaskExceptionBySqeId(aicpuComm, exceptionInfo.sqId, sqeId);
    CHK_PRT_CONT(ret != HCCL_SUCCESS, HCCL_ERROR("[PrintTaskExceptionBySqeId]fail, ret[%d], group[%s], sqId[%u], taskId[%u]",
        ret, aicpuComm->GetIdentifier().c_str(), exceptionInfo.sqId, exceptionInfo.taskId)); // 如果上报失败，继续打印taskException

    ret = ReportErrMsg(aicpuComm, exceptionInfo);
    CHK_PRT_CONT(ret != HCCL_SUCCESS, HCCL_ERROR("[ReportErrMsg]fail, ret[%d], group[%s], sqId[%u], taskId[%u]",
        ret, aicpuComm->GetIdentifier().c_str(), exceptionInfo.sqId, exceptionInfo.taskId)); // 如果上报失败，继续打印taskException

    // notify超时场景：step1 打印当前流信息；step2 打印当前通信域信息；step3 打印其他通信域信息
    if (cqeStatus == dfx::CqeStatus::kCqeException && exceptionInfo.sqeType == RT_STARS_SQE_TYPE_PLACE_HOLDER) {
        CHK_RET(PrintCommTaskException(aicpuComm));
        for (auto &commInfo : aicpuCommInfo) {
            CollCommAicpu *comm = commInfo.second->GetCollCommAicpu();
            if (comm != nullptr && comm->GetIdentifier() != aicpuComm->GetIdentifier()) {
                CHK_RET(PrintCommTaskException(comm));
            }
        }
    }
    return ret;
}

u32 HcclCommTaskExceptionLite::GetSqeId(uint16_t taskId, uint16_t streamId)
{
    return (static_cast<u32>(taskId) << 16) | static_cast<u32>(streamId);
}
 	 
HcclResult HcclCommTaskExceptionLite::ReportErrMsg(CollCommAicpu *aicpuComm, const rtLogicCqReport_t &exceptionInfo)
{
    CHK_PTR_NULL(aicpuComm);
    CHK_PTR_NULL(aicpuComm->GetHcclCommDfxLite());
    CHK_PTR_NULL(aicpuComm->GetHcclCommDfxLite()->GetMirrorTaskManagerLite());

    const u32 sqeId = GetSqeId(exceptionInfo.taskId, exceptionInfo.streamId);
    HCCL_INFO("[%s]group[%s], sqeId[0x%x], taskId[%u], streamId[%u].",
        __func__, aicpuComm->GetIdentifier().c_str(), sqeId, exceptionInfo.taskId, exceptionInfo.streamId);

    const auto curTask = aicpuComm->GetHcclCommDfxLite()->GetMirrorTaskManagerLite()->GetTaskInfo(exceptionInfo.sqId, sqeId);
    CHK_SMART_PTR_NULL(curTask);
    CHK_SMART_PTR_NULL(curTask->dfxOpInfo_);

    if (!aicpuComm->IsErrorReported()) {
        // 1) errorMessage上报
        Hccl::ErrorMessageReport errMsgInfo{};
        CHK_RET(GenerateErrorMessageReport(aicpuComm, *curTask, exceptionInfo, errMsgInfo));
        CHK_RET(aicpuComm->SendErrorMessageReportToHost(errMsgInfo));

        // 2) send mbox to tsfw
        u32 notifyId = curTask->dfxOpInfo_->cpuWaitAicpuNotifyId_;
        CHK_RET(SendTaskExceptionByMBox(notifyId, 0, exceptionInfo));
        aicpuComm->SetErrorReported(true);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::PrintTaskExceptionBySqeId(CollCommAicpu *aicpuComm, u32 sqId, u32 sqeId)
{
    CHK_PTR_NULL(aicpuComm);
    CHK_PTR_NULL(aicpuComm->GetHcclCommDfxLite());
    CHK_PTR_NULL(aicpuComm->GetHcclCommDfxLite()->GetMirrorTaskManagerLite());

    const auto curTask = aicpuComm->GetHcclCommDfxLite()->GetMirrorTaskManagerLite()->GetTaskInfo(sqId, sqeId);
    CHK_SMART_PTR_NULL(curTask);
    CHK_SMART_PTR_NULL(curTask->dfxOpInfo_);

    // 已经打印过的不再重复打印
    auto it = threadsPrinted_.find(sqId);
    if (it != threadsPrinted_.end() && it->second == sqeId) {
        HCCL_RUN_INFO("TaskException][AICPU]sqId:%u, sqeId:%u has been printed, skip", sqId, sqeId);
        return HCCL_SUCCESS;
    }
    threadsPrinted_[sqId] = sqeId;

    u32 sqHead = 0U;
    u32 sqTail = 0U;
    (void)QuerySqStatus(devId_, sqId, sqHead, sqTail);

    // 1. 打印task信息
    HCCL_ERROR("[TaskException][AICPU]base information is %s, %s, sqHead:%u, sqTail:%u",
        curTask->GetIndopBaseInfo().c_str(), curTask->GetParaInfo().c_str(), sqHead, sqTail);
    // 2. UB任务打印EID信息
    PrintEid(*curTask);
    // 3. 打印group信息
    HCCL_ERROR("[TaskException][AICPU]group information is %s.", GetGroupInfo(aicpuComm).c_str());
    // 4. 打印算子信息和task序列
    if (curTask->taskParam_.taskType != Hccl::TaskParamType::TASK_NOTIFY_WAIT) { // 非notify场景，仅打印算子信息
        HCCL_ERROR("[TaskException][AICPU]opData information is %s.", curTask->GetIndopDataInfo().c_str());
    } else {
        CHK_RET(PrintTaskContextInfo(aicpuComm, sqId, sqeId)); // notify场景打印算子信息和task序列
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::GenerateErrorMessageReport(CollCommAicpu *aicpuComm,
    const Hccl::TaskInfo& taskInfo, const rtLogicCqReport_t &exceptionInfo, Hccl::ErrorMessageReport &errMsgInfo)
{
    // 获取需要上报的关键信息
    errMsgInfo.remoteUserRank = taskInfo.GetRemoteRankId();
    errMsgInfo.streamId = taskInfo.streamId_;
    errMsgInfo.taskId = taskInfo.taskId_;
    errMsgInfo.rankId = aicpuComm->GetTopoInfo().userRank;
    errMsgInfo.rankSize = aicpuComm->GetTopoInfo().userRankSize;
    errMsgInfo.opIndex = taskInfo.dfxOpInfo_->opIndex_;
    errMsgInfo.opType = taskInfo.dfxOpInfo_->op_.opType;
    errMsgInfo.count = taskInfo.dfxOpInfo_->op_.dataCount;
    errMsgInfo.dataType = taskInfo.dfxOpInfo_->op_.dataType;
    errMsgInfo.srcAddr = taskInfo.dfxOpInfo_->op_.newInputMem;
    errMsgInfo.dstAddr = taskInfo.dfxOpInfo_->op_.newOutputMem;
    errMsgInfo.taskType = taskInfo.taskParam_.taskType;

    errMsgInfo.rtCqErrorType = exceptionInfo.errorType;
    errMsgInfo.rtCqErrorCode = exceptionInfo.errorCode;

    CHK_SAFETY_FUNC_RET(memcpy_s(errMsgInfo.tag, sizeof(errMsgInfo.tag),
        taskInfo.dfxOpInfo_->algTag_.c_str(), taskInfo.dfxOpInfo_->algTag_.size()));
    CHK_SAFETY_FUNC_RET(memcpy_s(errMsgInfo.group, sizeof(errMsgInfo.group),
        aicpuComm->GetIdentifier().c_str(), aicpuComm->GetIdentifier().size()));

    GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);
    return HCCL_SUCCESS;
}

void HcclCommTaskExceptionLite::GenerateTaskErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport &errMsgInfo,
    const rtLogicCqReport_t &exceptionInfo)
{
    switch (taskInfo.taskParam_.taskType) {
        case Hccl::TaskParamType::TASK_NOTIFY_WAIT:
        case Hccl::TaskParamType::TASK_NOTIFY_RECORD:
            FillNotifyErrMsg(taskInfo, errMsgInfo);
            break;
        case Hccl::TaskParamType::TASK_UB_REDUCE_INLINE:
        case Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY:
            FillReduceErrMsg(taskInfo, errMsgInfo, exceptionInfo);
            break;
        case Hccl::TaskParamType::TASK_REDUCE_INLINE:
            FillReduceInlineErrMsg(taskInfo, errMsgInfo);
            break;
        case Hccl::TaskParamType::TASK_UB_INLINE_WRITE:
        case Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY:
            FillDmaErrMsg(taskInfo, errMsgInfo, exceptionInfo);
            break;
        case Hccl::TaskParamType::TASK_UB:
            FillUbErrMsg(taskInfo, errMsgInfo, exceptionInfo);
            break;
        case Hccl::TaskParamType::TASK_SDMA:
            FillSdmaErrMsg(taskInfo, errMsgInfo);
            break;
        default:
            HCCL_ERROR("[TaskException][AICPU]%s taskType[%d] is not support", __func__, taskInfo.taskParam_.taskType);
            return;
    }
}

void HcclCommTaskExceptionLite::FillNotifyErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport &errMsgInfo)
{
    errMsgInfo.notifyId = taskInfo.taskParam_.taskPara.Notify.notifyID;
    errMsgInfo.notifyValue = taskInfo.taskParam_.taskPara.Notify.value;
}

void HcclCommTaskExceptionLite::FillReduceErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport &errMsgInfo,
    const rtLogicCqReport_t &exceptionInfo)
{
    errMsgInfo.reduceType = taskInfo.taskParam_.taskPara.Reduce.reduceOp;
    errMsgInfo.notifyId = taskInfo.taskParam_.taskPara.Reduce.notifyID;
    errMsgInfo.notifyValue = taskInfo.taskParam_.taskPara.Reduce.notifyValue;
    errMsgInfo.locEid = taskInfo.taskParam_.taskPara.Reduce.locEid;
    errMsgInfo.rmtEid = taskInfo.taskParam_.taskPara.Reduce.rmtEid;
    errMsgInfo.ubCqeStatus = exceptionInfo.errorCode & 0xFF;
    errMsgInfo.linkType = taskInfo.taskParam_.taskPara.Reduce.linkType;
    errMsgInfo.size = taskInfo.taskParam_.taskPara.Reduce.size;
    errMsgInfo.taskSrcAddr = reinterpret_cast<u64>(taskInfo.taskParam_.taskPara.Reduce.src);
    errMsgInfo.taskDstAddr = reinterpret_cast<u64>(taskInfo.taskParam_.taskPara.Reduce.dst);
    HCCL_ERROR("[TaskException][AICPU]ubCqeStatus[%u], localEid[%s], remoteEid[%s]. ",
        errMsgInfo.ubCqeStatus, errMsgInfo.locEid.Describe().c_str(), errMsgInfo.rmtEid.Describe().c_str());
}

void HcclCommTaskExceptionLite::FillDmaErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport &errMsgInfo,
    const rtLogicCqReport_t &exceptionInfo)
{
    errMsgInfo.notifyId = taskInfo.taskParam_.taskPara.DMA.notifyID;
    errMsgInfo.notifyValue = taskInfo.taskParam_.taskPara.DMA.notifyValue;
    FillUbErrMsg(taskInfo, errMsgInfo, exceptionInfo);
}

void HcclCommTaskExceptionLite::FillSdmaErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport &errMsgInfo)
{
    errMsgInfo.linkType = taskInfo.taskParam_.taskPara.DMA.linkType;
    errMsgInfo.size = taskInfo.taskParam_.taskPara.DMA.size;
    errMsgInfo.taskSrcAddr = reinterpret_cast<u64>(taskInfo.taskParam_.taskPara.DMA.src);
    errMsgInfo.taskDstAddr = reinterpret_cast<u64>(taskInfo.taskParam_.taskPara.DMA.dst);
}

void HcclCommTaskExceptionLite::FillUbErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport &errMsgInfo,
    const rtLogicCqReport_t &exceptionInfo)
{
    errMsgInfo.locEid = taskInfo.taskParam_.taskPara.DMA.locEid;
    errMsgInfo.rmtEid = taskInfo.taskParam_.taskPara.DMA.rmtEid;
    errMsgInfo.ubCqeStatus = exceptionInfo.errorCode & 0xFF;
    errMsgInfo.linkType = taskInfo.taskParam_.taskPara.DMA.linkType;
    errMsgInfo.size = taskInfo.taskParam_.taskPara.DMA.size;
    errMsgInfo.taskSrcAddr = reinterpret_cast<u64>(taskInfo.taskParam_.taskPara.DMA.src);
    errMsgInfo.taskDstAddr = reinterpret_cast<u64>(taskInfo.taskParam_.taskPara.DMA.dst);
    HCCL_ERROR("[TaskException][AICPU]ubCqeStatus[%u], localEid[%s], remoteEid[%s]. ",
        errMsgInfo.ubCqeStatus, errMsgInfo.locEid.Describe().c_str(), errMsgInfo.rmtEid.Describe().c_str());
}

void HcclCommTaskExceptionLite::FillReduceInlineErrMsg(const Hccl::TaskInfo& taskInfo,
    Hccl::ErrorMessageReport &errMsgInfo)
{
    errMsgInfo.reduceType = taskInfo.taskParam_.taskPara.Reduce.reduceOp;
}

HcclResult HcclCommTaskExceptionLite::SendTaskExceptionByMBox(const u32 notifyId, const u32 tsId,
    const rtLogicCqReport_t &exceptionInfo)
{
    ts_aicpu_msg_info_t aicpuSqe = {};
    u32 hostpid = 0;
    u32 vfId = 0;
    int pid = getpid();
    HCCL_INFO("[%s]getpid[%d]", __func__, pid);
    // 调整drvQueryProcessHostPid获取pid和vf_id的值
    CHK_RET(HrtHalDrvQueryProcessHostPid(pid, nullptr, &vfId, &hostpid, nullptr));

    aicpuSqe.pid = hostpid;
    aicpuSqe.cmd_type = TS_AICPU_RECORD;
    aicpuSqe.vf_id = vfId;
    aicpuSqe.tid = 0U;  // notify is no need tid
    aicpuSqe.u.aicpu_record.record_type = AICPU_MSG_NOTIFY_RECORD_V2;
    aicpuSqe.u.aicpu_record.record_id = notifyId;
    aicpuSqe.ts_id = static_cast<uint8_t>(tsId);
    aicpuSqe.u.aicpu_record.fault_task_id = 0xffffffff;

    if (exceptionInfo.sqeType == ubSqeType) {
        aicpuSqe.u.aicpu_record.ret_code = SwitchUBCqeErrCodeToTsErrCode(exceptionInfo.errorCode & 0xFF);
    } else if (exceptionInfo.sqeType == sdmaSqeType) {
        aicpuSqe.u.aicpu_record.ret_code = SwitchSdmaCqeErrCodeToTsErrCode(exceptionInfo.errorCode);
    } else {
        aicpuSqe.u.aicpu_record.ret_code = TS_ERROR_HCCL_OTHER_ERROR;
    }

    struct event_summary event;
    event.dst_engine = TS_CPU;
    event.policy = ONLY;
    event.pid = 0;
    event.grp_id = 0;
    event.event_id = EVENT_TS_CTRL_MSG;
    event.subevent_id = 0U;
    event.msg_len = static_cast<uint32_t>(sizeof(ts_aicpu_msg_info_t));
    event.msg = reinterpret_cast<char_t *>(&aicpuSqe);
    drvError_t ret = Hccl::DlHalFunctionV2::GetInstance().dlHalEschedSubmitEvent(devId_, &event);
    if (ret != DRV_ERROR_NONE) {
        HCCL_ERROR("[%s]dlHalEschedSubmitEvent failed, ret=%d, notifyId=%u, hostpid=%u, vfId=%u, tsId=%u",
            __func__, ret, notifyId, hostpid, vfId, tsId);
        return HCCL_E_DRV;
    }
    HCCL_RUN_INFO("[%s]fininsh, notifyId=%u, hostpid=%u, vfId=%u, tsId=%u, errorType=%u, errorCode=%u, ret_code=%u",
        __func__, notifyId, hostpid, vfId, tsId, exceptionInfo.errorType, exceptionInfo.errorCode,
        aicpuSqe.u.aicpu_record.ret_code);
    return HCCL_SUCCESS;
}

// 把UB类错误码转换成Ts对应的错误码
uint16_t HcclCommTaskExceptionLite::SwitchUBCqeErrCodeToTsErrCode(u32 cqeErrCode) {
    switch (cqeErrCode) {
        case RT_UB_LOCAL_OPERATIOINERR:
            return TS_ERROR_HCCL_OP_UB_DDRC_FAILED;
        case RT_UB_REMOTE_OPERATIOINERR:
            return TS_ERROR_HCCL_OP_UB_POISON_FAILED;
        case RT_UB_LINK_FAILEDERR:
            return TS_ERROR_HCCL_OP_UB_LINK_FAILED;
        default:
            return TS_ERROR_HCCL_OTHER_ERROR;
    }
}

// 把SDMA类错误码转换成Ts对应的错误码
uint16_t HcclCommTaskExceptionLite::SwitchSdmaCqeErrCodeToTsErrCode(u32 cqeErrCode) {
    switch (cqeErrCode) {
        case RT_SDMA_COMPERR:
            return TS_ERROR_SDMA_LINK_ERROR;
        case RT_SDMA_COMPDATAERR:
            return TS_ERROR_SDMA_POISON_ERROR;
        case RT_SDMA_DATAERR:
            return TS_ERROR_SDMA_DDRC_ERROR;
        default:
            return TS_ERROR_HCCL_OTHER_ERROR;
    }
}

HcclResult HcclCommTaskExceptionLite::CollectTaskContext(CollCommAicpu *aicpuComm, u32 sqId, u32 taskId,
    std::vector<std::shared_ptr<Hccl::TaskInfo>> &taskContext)
{
    auto queue = aicpuComm->GetHcclCommDfxLite()->GetMirrorTaskManagerLite()->GetQueue(sqId);
    CHK_PRT_RET(queue == nullptr,
        HCCL_ERROR("[%s]GetQueue nullptr, devId[%u], sqId[%u].", __func__, devId_, sqId), HCCL_E_PARA);

    auto func = [taskId] (const std::unique_ptr<Hccl::TaskInfo>& task) { return task->taskId_ == taskId; };
    auto taskIterPtr = queue->Find(func);
    CHK_PRT_RET(taskIterPtr == nullptr || *taskIterPtr == *queue->End(),
        HCCL_ERROR("[%s]exception task not found, devId[%u], sqId[%u], taskId[%u]", __func__, devId_, sqId, taskId),
        HCCL_E_PARA);

    // 找到当前异常task的前50个task(至多)
    for (uint32_t i = 0; i < TASK_CONTEXT_SIZE && !queue->IsEmpty(); ++i, --(*taskIterPtr)) {
        if ((**taskIterPtr)->taskId_ > taskId) { // 回绕中止
            HCCL_ERROR("[%s]prev taskId[%u] is bigger than err taskId[%u], taskNum[%u], stop traversal",
                __func__, (**taskIterPtr)->taskId_, taskId, taskContext.size());
            break;
        }
        taskContext.emplace_back((**taskIterPtr).get());

        if (*taskIterPtr == *queue->Begin()) { // 遍历到起始位置中止
            HCCL_ERROR("[%s]taskId[%u] is queue Begin, taskNum[%u], stop traversal",
                __func__, (**taskIterPtr)->taskId_, taskContext.size());
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::PrintTaskContextInfo(CollCommAicpu *aicpuComm, u32 sqId, u32 taskId)
{
    std::vector<std::shared_ptr<Hccl::TaskInfo>> taskContext {};
    CHK_PRT_RET(CollectTaskContext(aicpuComm, sqId, taskId, taskContext) != HCCL_SUCCESS,
        HCCL_ERROR("[%s]CollectTaskContext failed, devId[%u], sqId[%u], taskId[%u]", __func__, devId_, sqId, taskId),
        HCCL_E_PARA);

    std::string taskContextInfo = "";
    Hccl::TaskInfo* lastTask = nullptr;
    for (u32 i = 0; i < taskContext.size(); ++i) {
        if (taskContext[i] == nullptr) {
            HCCL_ERROR("[%s]taskContext[%u] is nullptr, skip!", __func__, i);
            continue;
        }
        if (lastTask == nullptr) {
            lastTask = taskContext[i].get();
        }
        std::string conciseInfo = taskContext[i]->GetConciseBaseInfo() + ",";

        u32 lastOpIndex = ((lastTask->dfxOpInfo_ == nullptr) ? UINT32_MAX : lastTask->dfxOpInfo_->opIndex_);
        u32 curOpIndex = ((taskContext[i]->dfxOpInfo_ == nullptr) ? UINT32_MAX : taskContext[i]->dfxOpInfo_->opIndex_);
        bool overSize = (taskContextInfo.size() + conciseInfo.size()) >= TASK_CONTEXT_INFO_SIZE; // 1. 字符串超过一定长度时，打印一次
        // 2. 不同算子，新起一行打印
        if (overSize || (lastOpIndex != curOpIndex)) {
            HCCL_ERROR("[TaskException][AICPU]opData information is %s.", lastTask->GetIndopDataInfo().c_str());
            HCCL_ERROR("[TaskException][AICPU]task sequence is OP(%u): %s", lastOpIndex, taskContextInfo.c_str());
            taskContextInfo = "";
            lastTask = taskContext[i].get();
        }
        taskContextInfo += conciseInfo;
    }

    // 3. 最后一个task，打印一次
    if (!taskContextInfo.empty() && lastTask != nullptr) {
        u32 lastOpIndex = (lastTask->dfxOpInfo_ == nullptr) ? UINT32_MAX : lastTask->dfxOpInfo_->opIndex_;
        HCCL_ERROR("[TaskException][AICPU]opData information is %s.", lastTask->GetIndopDataInfo().c_str());
        HCCL_ERROR("[TaskException][AICPU]task sequence is OP(%u): %s", lastOpIndex, taskContextInfo.c_str());
    }
    HCCL_ERROR("[TaskException][AICPU]task sequence end.");
    return HCCL_SUCCESS;
}

std::string HcclCommTaskExceptionLite::GetGroupInfo(CollCommAicpu *aicpuComm)
{
    if (aicpuComm == nullptr) {
        HCCL_ERROR("[%s]aicpuComm is nullptr, return empty string.", __func__);
        return "";
    }
    return Hccl::StringFormat("group:[%s], rankSize:[%u], localRank:[%d]",
	    aicpuComm->GetIdentifier().c_str(), aicpuComm->GetTopoInfo().userRankSize, aicpuComm->GetTopoInfo().userRank);
}

void HcclCommTaskExceptionLite::PrintEid(const Hccl::TaskInfo& taskInfo)
{
    if (taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB_REDUCE_INLINE ||
        taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY) {
        HCCL_ERROR("[TaskException][AICPU][%s]Error UB link info: localEid[%s], remoteEid[%s].", __func__,
            taskInfo.taskParam_.taskPara.Reduce.locEid.Describe().c_str(),
            taskInfo.taskParam_.taskPara.Reduce.rmtEid.Describe().c_str());
    } else if (taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB_INLINE_WRITE ||
        taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY ||
        taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB) {
        HCCL_ERROR("[TaskException][AICPU][%s]Error UB link info: localEid[%s], remoteEid[%s].", __func__,
            taskInfo.taskParam_.taskPara.DMA.locEid.Describe().c_str(),
            taskInfo.taskParam_.taskPara.DMA.rmtEid.Describe().c_str());
    }
}
}