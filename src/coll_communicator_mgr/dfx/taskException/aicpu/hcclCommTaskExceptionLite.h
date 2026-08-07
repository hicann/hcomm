/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_COMM_TASKEXCEPTION_LIFT_H
#define HCCL_COMM_TASKEXCEPTION_LIFT_H

#include "daemon_func.h"
#include "mirror_task_manager.h"
#include "coll_comm_aicpu.h"
#include "aicpu_hccl_sqcq.h"
#include "error_message_v2.h"
#include "aicpu_indop_process.h"

namespace hcomm {

class HcclCommTaskExceptionLite : public Hccl::DaemonFunc {
public:
    static HcclCommTaskExceptionLite& GetInstance();
    void Init(u32 devId);
    void Call() override;
    HcclResult PrintAllCommTaskException(); // 打印所有通信域所有流的task信息
    HcclResult PrintCommTaskException(CollCommAicpu* aicpuComm);

private:
    HcclCommTaskExceptionLite() = default;
    ~HcclCommTaskExceptionLite() override = default;

    // 检测流上的异常cqe并进行打印和上报
    HcclResult HandleExceptionCqe();
    HcclResult GetThreadCqe(hccl::Thread* thread, rtLogicCqReport_t& cqeException, CqeStatus& cqeStatus);
    HcclResult ProcessCqe(
        CollCommAicpu* aicpuComm, const rtLogicCqReport_t& exceptionInfo, const CqeStatus& cqeStatus,
        const std::vector<std::pair<std::string, CollCommAicpuMgr*>>& aicpuCommInfo);

    // errMsg上报到host
    HcclResult ReportErrMsg(CollCommAicpu* aicpuComm, const rtLogicCqReport_t& exceptionInfo);
    HcclResult GenerateErrorMessageReport(
        CollCommAicpu* aicpuComm, const Hccl::TaskInfo& taskInfo, const rtLogicCqReport_t& exceptionInfo,
        Hccl::ErrorMessageReport& errMsgInfo);
    void GenerateTaskErrMsg(
        const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t& exceptionInfo);
    void FillNotifyErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo);
    void FillReduceErrMsg(
        const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t& exceptionInfo);
    void FillDmaErrMsg(
        const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t& exceptionInfo);
    void FillSdmaErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo);
    void FillUbErrMsg(
        const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t& exceptionInfo);
    void FillReduceInlineErrMsg(const Hccl::TaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo);
    HcclResult SendTaskExceptionByMBox(const u32 notifyId, const u32 tsId, const rtLogicCqReport_t& exceptionInfo);
    uint16_t SwitchUBCqeErrCodeToTsErrCode(u32 cqeErrCode);
    uint16_t SwitchSdmaCqeErrCodeToTsErrCode(u32 cqeErrCode);

    // 打印流上的task信息的方法函数
    HcclResult PrintTaskExceptionBySqeId(CollCommAicpu* aicpuComm, u32 sqId, u32 sqeId);
    HcclResult PrintTaskContextInfo(CollCommAicpu* aicpuComm, u32 sqId, u32 taskId);
    HcclResult
    CollectTaskContext(CollCommAicpu* aicpuComm, u32 sqId, u32 taskId, std::vector<Hccl::TaskInfo*>& taskContext);
    void PrintEid(const Hccl::TaskInfo& taskInfo);
    std::string GetGroupInfo(CollCommAicpu* aicpuComm);
    u32 GetSqeId(uint16_t taskId, uint16_t streamId);

    // dpu相关
    HcclResult HandleDpuTaskexception(CollCommAicpu* aicpuComm);
    HcclResult IsHandleDpuStop(uint8_t* taskexceptionVa, bool& isStop);

private:
    bool stopCall_{false}; // 避免taskException失败后刷屏
    u32 devId_{INVALID_UINT};
    Hccl::MirrorTaskManager* mirrorTaskManager_{nullptr}; // 使用原始指针，不管理生命周期
    std::unordered_map<u32, u32> threadsPrinted_; // sqId -> sqeId, 记录已经打印过taskException的流信息
};

} // namespace hcomm

#endif
