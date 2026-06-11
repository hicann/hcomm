/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hcclCommTaskException.h"
#include <memory>
#include "log.h"
#include "coll_comm.h"
#include "acl/acl_rt.h"
#include "orion_adapter_hccp.h"
#include <adapter_error_manager_pub.h>
#include "op_type.h"
#include "task_exception_handler.h"
#include "ccuTaskException.h"
#include "hccl_types.h"

namespace hcomm {

using namespace std;

constexpr u32 MAX_MODULE_DEVICE_NUM_V2 = 65;
constexpr uint32_t TASK_CONTEXT_SIZE = 50;
constexpr uint32_t TASK_CONTEXT_INFO_SIZE = LOG_TMPBUF_SIZE - 50; // task 执行失败时打印前序task信息的长度限制

std::mutex g_communicatorCallbackMapMutexV2;
array<map<s32, GetAicpuTaskExceptionCallBackHcomm>, MAX_MODULE_DEVICE_NUM_V2> g_communicatorCallbackMapV2;
std::mutex g_commHadCallbackArrayMutexV2;
array<bool, MAX_MODULE_DEVICE_NUM_V2> g_commHadCallbackArrayV2 = {false};


GetAicpuCqeErrInfoCallBackHcomm g_getAicpuCqeErrInfoCallBack = nullptr;
AicpuGetErrStatusVecCallBack g_AicpuGetErrStatusVecCallBack = nullptr;

void RegisterGetAicpuCqeErrInfoCallBackHcomm(GetAicpuCqeErrInfoCallBackHcomm p1)
{
    g_getAicpuCqeErrInfoCallBack = p1;
    return;
}

void TaskExceptionHost::ClusterMoniterGetAicpuCqeErrInfo(u32 remoteLocalId, u32 locDeviceId, uint16_t status, string localEid, string remoteEid, string remoteInsId)
{
    if (g_getAicpuCqeErrInfoCallBack != nullptr) {
        g_getAicpuCqeErrInfoCallBack(remoteLocalId, locDeviceId, status, localEid, remoteEid, remoteInsId);
    } else {
        HCCL_RUN_WARNING("[ClusterMoniterGetAicpuCqeErrInfo]g_getAicpuCqeErrInfoCallBack is nullptr.");
    }
    return;
}

void RegisterAicpuGetErrStatusVecCallBack(AicpuGetErrStatusVecCallBack p1)
{
    g_AicpuGetErrStatusVecCallBack = p1;
    return;
}

std::vector<std::string> AicpuGetErrStatusVec(s32 deviceLogicID)
{
    if (g_AicpuGetErrStatusVecCallBack != nullptr) {
        return g_AicpuGetErrStatusVecCallBack(deviceLogicID);
    } else {
        HCCL_RUN_WARNING("[AicpuGetErrStatusVec]g_AicpuGetErrStatusVecCallBack is nullptr.");
    }
    return std::vector<std::string>();
}

std::string AicpuGetAndPrintClusterMonitorErr(const rtExceptionInfo *exceptionInfo)
{
    auto errStatusVec = AicpuGetErrStatusVec(exceptionInfo->deviceid);
    std::string errMsg = "";
    int errSize = errStatusVec.size();
    if (errSize > 0) {
        int maxListSize = 3;  // 放入errMsg中的异常事件最多只有3个
        if (errSize <= maxListSize) {
            errMsg = "\nthere are(is) " + std::to_string(errSize) + " abnormal device(s):\n";
        } else {
            errMsg = "\nthere are " + std::to_string(errSize) + " abnormal device(s), " +
                "only the first 3 devices are listed:\n";
        }

        for (int i = 0; i < errSize; i++) {
            HCCL_ERROR("%s", errStatusVec[i].c_str());
            if (i < maxListSize) {
                errMsg += ("\t" + errStatusVec[i] + "\n");
            }
        }
    }
    return errMsg;
}

TaskExceptionHost::~TaskExceptionHost()
{
    (void)UnRegister();
}

HcclResult TaskExceptionHost::Register()
{
    CHK_PRT_RET(isRegistered_, HCCL_DEBUG("[%s]has been registered, skip", __func__), HCCL_SUCCESS);
    aclError ret = aclrtSetExceptionInfoCallback(Process);
    CHK_PRT_RET(ret != ACL_SUCCESS,
        HCCL_ERROR("[%s]aclrtSetExceptionInfoCallback failed, ret = [%u]", __func__, ret), HCCL_E_RUNTIME);
    isRegistered_ = true;
    HCCL_INFO("[TaskExceptionHost] registered success.");
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHost::UnRegister()
{
    aclError ret = aclrtSetExceptionInfoCallback(nullptr);
    CHK_PRT_RET(ret != ACL_SUCCESS,
        HCCL_ERROR("[%s]aclrtSetExceptionInfoCallback failed, ret[%u]", __func__, ret), HCCL_E_RUNTIME);
    isRegistered_ = false;
    HCCL_INFO("[TaskExceptionHost]%s success.", __func__);
    return HCCL_SUCCESS;
}

TaskExceptionHost *TaskExceptionHostManager::GetHandler(size_t devId)
{
    // 检查 devId 是否越界
    if (devId >= MAX_MODULE_DEVICE_NUM_V2) {
        HCCL_ERROR("[TaskExceptionHost][GetInstance] deviceLogicID[%lu] is invalid", devId);
        return nullptr;
    }

    static TaskExceptionHost handlers_[MAX_MODULE_DEVICE_NUM_V2];
    return &handlers_[devId];
}
TaskExceptionHostManager::TaskExceptionHostManager() {}

TaskExceptionHostManager::~TaskExceptionHostManager() {}

void TaskExceptionHostManager::RegisterGetAicpuTaskExceptionCallBack(s32 streamId, u32 deviceLogicId,
    GetAicpuTaskExceptionCallBackHcomm p1)
{
    if (deviceLogicId >= MAX_MODULE_DEVICE_NUM_V2) {
        HCCL_ERROR("[RegisterGetAicpuTaskExceptionCallBack] deviceLogicId[%u] out of range, max is %u",
            deviceLogicId, MAX_MODULE_DEVICE_NUM_V2 - 1);
        return;
    }
    lock_guard<mutex> lock(g_communicatorCallbackMapMutexV2);
    g_communicatorCallbackMapV2[deviceLogicId][streamId] = p1;
    return;
}

void TaskExceptionHostManager::UnregisterGetAicpuTaskExceptionCallBack(s32 streamId, u32 deviceLogicId)
{
    if (deviceLogicId >= MAX_MODULE_DEVICE_NUM_V2) {
        HCCL_ERROR("[UnregisterGetAicpuTaskExceptionCallBack] deviceLogicId[%u] out of range, max is %u",
            deviceLogicId, MAX_MODULE_DEVICE_NUM_V2 - 1);
        return;
    }
    lock_guard<mutex> lock(g_communicatorCallbackMapMutexV2);
    auto& deviceMap = g_communicatorCallbackMapV2[deviceLogicId];
    auto it = deviceMap.find(streamId);
    if (it != deviceMap.end()) {
        deviceMap.erase(it);
    }
    return;
}

HcclResult TaskExceptionHost::PrintUbRegisters(s32 devLogicId, RdmaHandle rdmaHandle)
{
    HCCL_INFO("[PrintUbRegister] start, devLogicId[%d], rdmaHandle[%p]", devLogicId, rdmaHandle);
    Hccl::AuxInfoIn in;
    in.cqe.status = 0xffffffff; // 0xffffffff代表查询所有寄存器
    in.auxInfoInType = Hccl::AuxInfoInType::AUX_INFO_IN_TYPE_CQE;
    in.cqe.sR = 0;
    Hccl::AuxInfoOut auxInfo;
    auto ret = Hccl::RaGetAuxInfo(rdmaHandle, in, auxInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[PrintUbRegister]GetUbRegisterInfo failed, devLogicId[%d], rdmaHandle[%p]", devLogicId, rdmaHandle);
        return ret;
    }

    uint16_t isAuxInfoExisted{false};
    for (u32 i = 0; i < auxInfo.auxInfoNum; i++) {
        if (auxInfo.auxInfoValues[i]) { // 非零进行打印
            isAuxInfoExisted = true;
            HCCL_ERROR("devLogicId[%d], cqe_aux_info_type[%u], cqe_aux_info_value[0x%x]",
 	  	            devLogicId, auxInfo.auxInfoTypes[i], auxInfo.auxInfoValues[i]);
 	    } else {
 	        HCCL_INFO("devLogicId[%d], cqe_aux_info_type[%u], cqe_aux_info_value[0x%x]",
 	            devLogicId, auxInfo.auxInfoTypes[i], auxInfo.auxInfoValues[i]);
        }
    }
    if (!isAuxInfoExisted) {
        HCCL_ERROR("devLogicId[%d], all aux_info values are zero.", devLogicId);
    }
    return HCCL_SUCCESS;
}

bool IsMC2Exception(rtExceptionInfo_t* exceptionInfo)
{
    return exceptionInfo->expandInfo.type == RT_EXCEPTION_FUSION &&
           exceptionInfo->expandInfo.u.fusionInfo.type == RT_FUSION_AICORE_CCU;
}

void TaskExceptionHost::Process(rtExceptionInfo_t* exceptionInfo)
{
    if (exceptionInfo == nullptr) {
        HCCL_ERROR("[%s]fail, exceptionInfo is nullptr", __func__);
        return;
    }
    HCCL_RUN_INFO("[TaskExceptionHost][%s], taskid[%u], streamid[%u], tid[%u], deviceid[%u], retcode[%u], type[%d]",
        __func__, exceptionInfo->taskid, exceptionInfo->streamid, exceptionInfo->tid, exceptionInfo->deviceid,
        exceptionInfo->retcode, exceptionInfo->expandInfo.type);

    if (IsMC2Exception(exceptionInfo)) { // MC2 taskException 新流程暂未支持，回退到老流程
        Hccl::TaskExceptionHandler::Process(exceptionInfo);
        return;
    }

    std::shared_ptr<Hccl::TaskInfo> curTask = nullptr;
    HcclResult ret = Hccl::GlobalMirrorTasks::Instance().FindTaskInfo(exceptionInfo->deviceid, exceptionInfo->streamid,
        exceptionInfo->taskid, curTask);
    CHK_PRT_RET(ret == HCCL_E_NOT_FOUND, HCCL_RUN_WARNING("[%s]FindTaskInfo not found, deviceid[%u] streamid[%u] taskid[%u].",
        __func__, exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid),);

    CHK_PRT_RET(curTask == nullptr || ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]FindTaskInfo fail, curTask[%p], ret[%d], deviceid[%u], streamid[%u], taskid[%u].",
            __func__, curTask.get(), ret, exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid),);

    if (curTask->dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[%s]fail, dfxOpInfo is nullptr", __func__);
        return;
    }

    bool isIndop_ = curTask->dfxOpInfo_->isIndop_;
    HCCL_INFO("[%s]isIndop_[%d], taskType[%s]", __func__, isIndop_, curTask->taskParam_.taskType.Describe().c_str());
    if (!isIndop_) {
        Hccl::TaskExceptionHandler::Process(exceptionInfo);
        return;
    }

    if (curTask->taskParam_.taskType == Hccl::TaskParamType::TASK_CCU) {
        CcuTaskException::ProcessCcuException(exceptionInfo, *curTask); 
    } else {
        ProcessException(exceptionInfo, *curTask);
    }
}

std::string TaskExceptionHost::GetGroupRankInfo(const Hccl::TaskInfo& taskInfo)
{
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]TaskInfo communicator is nullptr.", __func__);
        return "";
    }

    hccl::CollComm *communicator = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
    return Hccl::StringFormat("group:[%s], rankSize[%u], rankId[%d]",
        communicator->GetCommId().c_str(), communicator->GetRankSize(), communicator->GetMyRankId());
}

void TaskExceptionHost::GetAicpuCqeErrRemoteLocalIdByRankId(hccl::CollComm* collComm, uint32_t rankid, u32 &remoteLocalId)
{
    if (collComm == nullptr || rankid == INVALID_VALUE_RANKID) {
        HCCL_ERROR("[GetAicpuCqeErrRemoteLocalIdByRankId]collComm is nullptr or rankId is invalid, rankId[%u]", rankid);
        remoteLocalId = INVALID_VALUE_RANKID;
        return;
    }

    Hccl::HcclCommunicator *commV2 = static_cast<Hccl::HcclCommunicator *>(collComm->GetCommunicatorV2());
    if (commV2 == nullptr) {
        HCCL_ERROR("[GetAicpuCqeErrRemoteLocalIdByRankId]commV2 is nullptr, rankId[%u]", rankid);
        remoteLocalId = INVALID_VALUE_RANKID;
        return;
    }
    void *rankGraph = nullptr;
    HcclResult ret =commV2->GetRankGraphV2(rankGraph);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetAicpuCqeErrRemoteLocalIdByRankId]GetRankGraphV2 failed, rankId[%u], ret[%d]", rankid, ret);
        remoteLocalId = INVALID_VALUE_RANKID;
        return;
    }
    Hccl::RankGraph *rankGraphv2 = static_cast<Hccl::RankGraph *>(rankGraph);
    u32 LocalId = rankGraphv2->GetLocalId(rankid);
    remoteLocalId = LocalId;
    return;
}

void TaskExceptionHost::GetAicpuCqeErrNetInstanceByRankId(hccl::CollComm* collComm, uint32_t rankid, std::string &netInstanceId)
{
    if (collComm == nullptr || rankid == INVALID_VALUE_RANKID) {
        HCCL_ERROR("[GetAicpuCqeErrNetInstanceByRankId]collComm is nullptr or rankId is invalid, rankId[%u]", rankid);
        netInstanceId = "";
        return;
    }
    
    Hccl::HcclCommunicator *commV2 = static_cast<Hccl::HcclCommunicator *>(collComm->GetCommunicatorV2());
    if (commV2 == nullptr) {
        HCCL_ERROR("[GetAicpuCqeErrNetInstanceByRankId]commV2 is nullptr, rankId[%u]", rankid);
        netInstanceId = "";
        return;
    }
    void *rankGraph = nullptr;
    HcclResult ret = commV2->GetRankGraphV2(rankGraph);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetAicpuCqeErrNetInstanceByRankId]GetRankGraphV2 failed, rankId[%u], ret[%d]", rankid, ret);
        netInstanceId = "";
        return;
    }
    Hccl::RankGraph *rankGraphv2 = static_cast<Hccl::RankGraph *>(rankGraph);
    const Hccl::NetInstance *netInstance = rankGraphv2->GetNetInstanceByRankId(0, rankid);
    if (netInstance == nullptr) {
        HCCL_ERROR("[GetAicpuCqeErrNetInstanceByRankId]netInstance is nullptr, rankId[%u]", rankid);
        netInstanceId = "";
        return;
    }
    std::string netInsId = netInstance->GetNetInstId();
    netInstanceId = netInsId;
    return;
}

void TaskExceptionHost::GetAicpuCqeErrInfo(rtExceptionInfo_t* exceptionInfo, const Hccl::ErrorMessageReport &errorMessage, const Hccl::TaskInfo& taskInfo)
{
    hccl::CollComm *collComm = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
    u32 remoteLocalId = INVALID_VALUE_RANKID;
    GetAicpuCqeErrRemoteLocalIdByRankId(collComm, errorMessage.remoteUserRank, remoteLocalId);
    std::string netInstanceId = "";
    GetAicpuCqeErrNetInstanceByRankId(collComm, errorMessage.remoteUserRank, netInstanceId);
    ClusterMoniterGetAicpuCqeErrInfo(remoteLocalId, exceptionInfo->deviceid, errorMessage.ubCqeStatus, errorMessage.locEid.Describe(), errorMessage.rmtEid.Describe(), netInstanceId); // 上报AICPU CQE错误信息到集群监控
    return;
}

void TaskExceptionHost::ProcessException(rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo)
{
    HCCL_RUN_INFO("[TaskExceptionHost][%s]begin to execute hccl task exception callback function.", __func__);
    bool isExistAicpuError = false;
    if (exceptionInfo == nullptr) {
        HCCL_ERROR("[TaskExceptionHost][ProcessException] exceptionInfo is nullptr.");
        return;
    }
    PrintAicpuErrorMessage(exceptionInfo, taskInfo, isExistAicpuError);
    if (isExistAicpuError) {
        // 如果已经有AICPU上报的task exception, 则host侧无需再次重复上报
        return;
    }
    HCCL_ERROR("[TaskExceptionHost][%s]Task from HCCL run failed.", __func__);
    if (taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_NOTIFY_WAIT) {
        PrintTaskContextInfo(exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid);
        HCCL_ERROR("[TaskExceptionHost][ProcessException] EI0002");
        RPT_INPUT_ERR(true,
            "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>({
                std::to_string(taskInfo.remoteRank_),
                taskInfo.GetBaseInfo(), (taskInfo.GetParaInfo()),
                ""})
        );
    }
    std::string clusterMonitorErrMsg = AicpuGetAndPrintClusterMonitorErr(exceptionInfo);
    HCCL_ERROR("[TaskExceptionHost]Task run failed, base information is deviceID:[%u], %s.",
        exceptionInfo->deviceid, taskInfo.GetIndopBaseInfo().c_str());
    HCCL_ERROR("[TaskExceptionHost]Task run failed, para information is %s.", taskInfo.GetParaInfo().c_str());
    HCCL_ERROR("[TaskExceptionHost]Task run failed, groupRank information is %s.",
        GetGroupRankInfo(taskInfo).c_str());
    HCCL_ERROR("[TaskExceptionHost]Task run failed, opData information is %s.", taskInfo.GetIndopDataInfo().c_str());
}

void TaskExceptionHost::PrintTaskContextInfo(uint32_t deviceId, uint32_t streamId, uint32_t taskId)
{
    Hccl::TaskInfoQueue *queue = nullptr;
    try {
        queue = Hccl::GlobalMirrorTasks::Instance().GetQueue(deviceId, streamId);
    } catch (Hccl::HcclException &e) {
        HCCL_ERROR("Exception task queue  not found. deviceId[%u], streamId[%u].", deviceId, streamId);
        return ;
    }

    if (queue == nullptr) {
        // 未找到异常对应的TaskQueue
        HCCL_ERROR("Exception task queue not found. deviceId[%u], streamId[%u].", deviceId, streamId);
        return;
    }

    auto func = [taskId] (const shared_ptr<Hccl::TaskInfo>& task) { return task->taskId_ == taskId; };
    auto taskIterPtr = queue->Find(func);
    if (taskIterPtr == nullptr || *taskIterPtr == *queue->End()) {
        // 在队列中未找到异常对应的TaskInfo
        HCCL_ERROR("Exception task not found. deviceId[%u], streamId[%u], taskId[%u].", deviceId, streamId, taskId);
        return;
    }

    // 找到当前异常task的前50个task(至多)
    vector<shared_ptr<Hccl::TaskInfo>> taskContext {};
    for (uint32_t i = 0; i < TASK_CONTEXT_SIZE && *taskIterPtr != *queue->Begin(); ++i, --(*taskIterPtr)) {
        if ((**taskIterPtr)->taskId_ > taskId) {
            HCCL_ERROR("[%s]prev taskId[%u]is bigger than err taskId[%u], traversal end.",
                __func__, (**taskIterPtr)->taskId_, taskId);
            break;
        }
        taskContext.emplace_back(**taskIterPtr);
    }

    HCCL_ERROR("[TaskExceptionHost]Task run failed, context sequence before error task is "
        "[SDMA:M(rank), RDMA:RS(rank,id), SendPayload:SP(rank), InlineReduce:IR(rank), Reduce:R(rank), "
        "NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), SendNotify:SN(rank,id), "
        "WriteWithNotify:WN(rank,id), WriteReduceWithNotify:WRN(rank,id)]:");

    std::string taskContextInfo = "";
    for (auto it = taskContext.rbegin(); it != taskContext.rend(); ++it) {
        std::string conciseInfo = (*it)->GetConciseBaseInfo();
        conciseInfo += ",";

        if (taskContextInfo.size() + conciseInfo.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("[TaskExceptionHost]%s", taskContextInfo.c_str());
            taskContextInfo = "";
        }

        taskContextInfo += conciseInfo;
    }
    HCCL_ERROR("[TaskExceptionHost]%s end.", taskContextInfo.c_str());
}
    
 	 
inline void PrintBaseErrorLog(const std::string &stageErrInfo, const std::string &baseInfo)
{
    HCCL_ERROR("%sTask run failed, base information is %s", stageErrInfo.c_str(), baseInfo.c_str());
}

inline void PrintParaErrorLog(const std::string &stageErrInfo, const std::string &paraInfoStr)
{
    HCCL_ERROR("%sTask run failed, para information is %s.", stageErrInfo.c_str(), paraInfoStr.c_str());
}

inline void PrintOpDataErrorLog(const std::string &stageErrInfo, const std::string &opDataContent)
{
    HCCL_ERROR("%sTask run failed, opData information is %s", stageErrInfo.c_str(), opDataContent.c_str());
}

inline void PrintGroupErrorLog(const std::string &stageErrInfo, const std::string &groupRankContent)
{
    HCCL_ERROR("%sTask run failed, groupRank information is %s.", stageErrInfo.c_str(), groupRankContent.c_str());
}

void TaskExceptionHost::PrintGroupErrorMessage(Hccl::ErrorMessageReport &errorMessage, Hccl::TaskInfo &exceptionTaskInfo,
    std::string &groupRankContent, std::string &stageErrInfo)
{
    groupRankContent += "group:[";
    groupRankContent += std::string(errorMessage.group);
    groupRankContent += "], rankSize[";
    groupRankContent += std::to_string(errorMessage.rankSize);
    groupRankContent += "], localRank[";
    groupRankContent += std::to_string(errorMessage.rankId);
    groupRankContent += "], remoteRank[";
    groupRankContent += std::to_string(errorMessage.remoteUserRank);
    groupRankContent += "]";

    PrintGroupErrorLog(stageErrInfo, groupRankContent);
    return;
}

const std::map<HcclReduceOp, std::string> HCOM_REDUCE_OP_STR_MAP{
    {HcclReduceOp::HCCL_REDUCE_SUM, "sum"},
    {HcclReduceOp::HCCL_REDUCE_PROD, "prod"},
    {HcclReduceOp::HCCL_REDUCE_MAX, "max"},
    {HcclReduceOp::HCCL_REDUCE_MIN, "min"},
    {HcclReduceOp::HCCL_REDUCE_RESERVED, "invalid"}
};

inline std::string GetReduceOpEnumStr2(HcclReduceOp reduceOp)
{
    auto iter = HCOM_REDUCE_OP_STR_MAP.find(reduceOp);
    if (iter == HCOM_REDUCE_OP_STR_MAP.end()) {
        return "HcclReduceOp(" + std::to_string(reduceOp) + ")";
    } else {
        return iter->second;
    }
}

const std::map<HcclDataType, std::string> HCOM_DATA_TYPE_STR_MAP{
    {HcclDataType::HCCL_DATA_TYPE_INT8, "int8"},
    {HcclDataType::HCCL_DATA_TYPE_INT16, "int16"},
    {HcclDataType::HCCL_DATA_TYPE_INT32, "int32"},
    {HcclDataType::HCCL_DATA_TYPE_INT64, "int64"},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, "uint64"},
    {HcclDataType::HCCL_DATA_TYPE_FP16, "float16"},
    {HcclDataType::HCCL_DATA_TYPE_FP32, "float32"},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, "uint8"},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, "uint16"},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, "uint32"},
    {HcclDataType::HCCL_DATA_TYPE_FP64, "float64"},
    {HcclDataType::HCCL_DATA_TYPE_BFP16, "bfloat16"},
    {HcclDataType::HCCL_DATA_TYPE_INT128, "int128"},
    {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, "fp8e4m3"},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, "fp8e5m2"},
    {HcclDataType::HCCL_DATA_TYPE_RESERVED, "reserved"}
};

inline std::string GetDataTypeEnumStr2(HcclDataType dataType)
{
    auto iter = HCOM_DATA_TYPE_STR_MAP.find(dataType);
    if (iter == HCOM_DATA_TYPE_STR_MAP.end()) {
        return "HcclDataType(" + std::to_string(dataType) + ")";
    } else {
        return iter->second;
    }
}

inline std::string GetDataTypeEnumStr(u32 dataType)
{
    auto hcclDataType = static_cast<HcclDataType>(dataType);
    return GetDataTypeEnumStr2(hcclDataType);
}
inline std::string GetOpTypeEnumStr(u32 opType)
{
 	Hccl::OpType hcclOpType = static_cast<Hccl::OpType::Value>(opType);
 	return hcclOpType.Describe();
}

void TaskExceptionHost::PrintOpDataErrorMessage(u32 deviceId, Hccl::ErrorMessageReport &errorMessage,
    std::string &stageErrInfo)
{
    std::stringstream opDataStr;
    opDataStr << "src" << "[0x"
            << std::hex << errorMessage.srcAddr << "], dst[0x"
            << std::hex << errorMessage.dstAddr << "], ";

    std::string opStr;
    if (errorMessage.reduceType != HcclReduceOp::HCCL_REDUCE_RESERVED) {
        opStr += "reduceType[";
        opStr += GetReduceOpEnumStr2(static_cast<HcclReduceOp>(errorMessage.reduceType));
        opStr += "], ";
    }

    std::string opDataContent;
    opDataContent += "deviceId:[";
    opDataContent += std::to_string(deviceId);
    opDataContent += "], index[";
    opDataContent += std::to_string(errorMessage.opIndex);
    opDataContent += "], count[";
    opDataContent += std::to_string(errorMessage.count);
    opDataContent += "], ";
    opDataContent += opStr;
    opDataContent += opDataStr.str();
    opDataContent += "dataType[";
    opDataContent += GetDataTypeEnumStr(errorMessage.dataType);
    opDataContent += "].";

    PrintOpDataErrorLog(stageErrInfo, opDataContent);
    return;
}

void TaskExceptionHost::ReportErrorMsg(const Hccl::TaskInfo &exceptionTaskInfo, const std::string &groupRankContent,
    const Hccl::ErrorMessageReport &errorMessage, rtExceptionInfo_t *exceptionInfo)
{
    HCCL_RUN_INFO("[ReportErrorMsg] start, taskType[%d]", exceptionTaskInfo.taskParam_.taskType);

    std::string clusterMonitorErrMsg = AicpuGetAndPrintClusterMonitorErr(exceptionInfo);

    if (exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_NOTIFY_WAIT) {
        HCCL_ERROR("[ReportErrorMsg] EI0002");
        RPT_INPUT_ERR(true,
            "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>({
                std::to_string(exceptionTaskInfo.remoteRank_),
                exceptionTaskInfo.GetIndopBaseInfo().c_str(), (exceptionTaskInfo.GetParaInfo() + clusterMonitorErrMsg).c_str(),
                ""})
        );
    } else if (exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY 
        || exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY
        || exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB_INLINE_WRITE
        || exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB_REDUCE_INLINE
        || exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB) {
        hccl::CollComm *collComm = static_cast<hccl::CollComm*>(exceptionTaskInfo.dfxOpInfo_->comm_);
        std::string localServerId = "";
        GetAicpuCqeErrNetInstanceByRankId(collComm, errorMessage.rankId, localServerId);
        u32 localDeviceId = INVALID_VALUE_RANKID;
        GetAicpuCqeErrRemoteLocalIdByRankId(collComm, errorMessage.rankId, localDeviceId);
        std::string remoteServerId = "";
        GetAicpuCqeErrNetInstanceByRankId(collComm, errorMessage.remoteUserRank, remoteServerId);
        u32 remoteDeviceId = INVALID_VALUE_RANKID;
        GetAicpuCqeErrRemoteLocalIdByRankId(collComm, errorMessage.remoteUserRank, remoteDeviceId);
        Hccl::IpAddress localAddr(errorMessage.locEid);
        Hccl::IpAddress remoteAddr(errorMessage.rmtEid);
        HCCL_ERROR("[ReportErrorMsg] EI0018");
        RPT_INPUT_ERR(true,
            "EI0018",
            std::vector<std::string>({"localServerId", "localDeviceId", "localDeviceIp", "remoteServerId", "remoteDeviceId", "remoteDeviceIp"}),
            std::vector<std::string>({
                localServerId, std::to_string(localDeviceId), localAddr.GetReverseEid().Describe().c_str(),
                remoteServerId, std::to_string(remoteDeviceId), remoteAddr.GetReverseEid().Describe().c_str()})
            );
    }
}

void GetTaskParam(Hccl::TaskParam &taskParam, const Hccl::ErrorMessageReport &errorMessage) {
    if (errorMessage.taskType == Hccl::TaskParamType::TASK_NOTIFY_WAIT) {
        taskParam.taskPara.Notify.notifyID = errorMessage.notifyId;
        taskParam.taskPara.Notify.value = errorMessage.notifyValue;
    } else if (errorMessage.taskType == Hccl::TaskParamType::TASK_UB_REDUCE_INLINE ||
        errorMessage.taskType == Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY) {
        taskParam.taskPara.Reduce.notifyID = errorMessage.notifyId;
        taskParam.taskPara.Reduce.notifyValue = errorMessage.notifyValue;
        taskParam.taskPara.Reduce.src = reinterpret_cast<void *>(errorMessage.taskSrcAddr);
 	    taskParam.taskPara.Reduce.dst = reinterpret_cast<void *>(errorMessage.taskDstAddr);
 	    taskParam.taskPara.Reduce.linkType = errorMessage.linkType;
 	    taskParam.taskPara.Reduce.size = errorMessage.size;
    } else if (errorMessage.taskType == Hccl::TaskParamType::TASK_UB_INLINE_WRITE ||
        errorMessage.taskType == Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY) {
        taskParam.taskPara.DMA.notifyID = errorMessage.notifyId;
        taskParam.taskPara.DMA.notifyValue = errorMessage.notifyValue;
        taskParam.taskPara.DMA.src = reinterpret_cast<void *>(errorMessage.taskSrcAddr);
 	    taskParam.taskPara.DMA.dst = reinterpret_cast<void *>(errorMessage.taskDstAddr);
 	    taskParam.taskPara.DMA.linkType = errorMessage.linkType;
 	    taskParam.taskPara.DMA.size = errorMessage.size;
    }
}


void TaskExceptionHost::PrintAicpuErrorMessage(rtExceptionInfo_t *exceptionInfo,
    const Hccl::TaskInfo& taskInfo, bool &isExistAicpuError)
{
    Hccl::ErrorMessageReport errorMessage;
    unique_lock<std::mutex> lock(g_commHadCallbackArrayMutexV2);
    if (g_commHadCallbackArrayV2[exceptionInfo->deviceid]) {
        // 防止同一个device上出现通信主流和kernel流均出现task exception时runtime调用两次callback
        // HDC通道信息不是读清，防止aicpu task exception重复上报
        HCCL_WARNING("aicpu error message been reported. deviceid[%u]", exceptionInfo->deviceid);
        return;
    }
    lock.unlock();
    if (g_communicatorCallbackMapV2[exceptionInfo->deviceid].find(exceptionInfo->streamid) !=\
        g_communicatorCallbackMapV2[exceptionInfo->deviceid].end()) {
        // 找到对应的通信域，并调用回调函数从HDC通道获取AICPU异常信息
        errorMessage = (g_communicatorCallbackMapV2[exceptionInfo->deviceid])[exceptionInfo->streamid]();
        if (strlen(errorMessage.tag) > 0) {
            isExistAicpuError = true;
            std::string groupRankContent;
            u32 streamId = static_cast<u32>(errorMessage.streamId);
            std::string tag = std::string(errorMessage.tag);
            Hccl::TaskParam taskParam{};
            taskParam.taskType = errorMessage.taskType;
            GetTaskParam(taskParam, errorMessage);
            std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = std::make_shared<Hccl::DfxOpInfo>();
            dfxOpInfo->tag_ = tag;
            dfxOpInfo->algType_ = errorMessage.algType;
            dfxOpInfo->comm_ = taskInfo.dfxOpInfo_ != nullptr ? taskInfo.dfxOpInfo_->comm_ : nullptr;
            Hccl::TaskInfo exceptionTaskInfo(streamId, errorMessage.taskId, errorMessage.remoteUserRank, taskParam, dfxOpInfo);
            auto logKeywordL2 = exceptionTaskInfo.taskParam_.taskType ==
                Hccl::TaskParamType::TASK_NOTIFY_WAIT ? Hccl::LOG_KEYWORDS_TIMEOUT : Hccl::LOG_KEYWORDS_RUN_FAILED;
            auto stageErrInfo = "[" + Hccl::LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + Hccl::LOG_KEYWORDS_AICPU + "]";
            HCCL_ERROR("%sTask from HCCL run failed.", stageErrInfo.c_str());
            // 防止tag字符串过长， 信息分开打印
            PrintBaseErrorLog(stageErrInfo, exceptionTaskInfo.GetIndopBaseInfo());
            PrintParaErrorLog(stageErrInfo, exceptionTaskInfo.GetParaInfo());
            PrintGroupErrorMessage(errorMessage, exceptionTaskInfo, groupRankContent, stageErrInfo);
            PrintOpDataErrorMessage(exceptionInfo->deviceid, errorMessage, stageErrInfo);
            HCCL_ERROR("errorMessage taskType[%s], rtCqErrorType[%u], rtCqErrorCode[%u]. ",
                errorMessage.taskType.Describe().c_str(), (u32)errorMessage.rtCqErrorType, errorMessage.rtCqErrorCode);

            // 打印UB DFX寄存器信息
            PrintUbDfxInfo(exceptionInfo, errorMessage);
            ReportErrorMsg(exceptionTaskInfo, groupRankContent, errorMessage, exceptionInfo);
            if (errorMessage.ubCqeStatus != 0) {
                GetAicpuCqeErrInfo(exceptionInfo, errorMessage, taskInfo);
            }
            lock.lock();
            g_commHadCallbackArrayV2[exceptionInfo->deviceid] = true;
        } else {
            HCCL_WARNING("PrintAicpuErrorMessage No Valid errorMessage!");
        }
    } else {
        HCCL_INFO("PrintAicpuErrorMessage streamId[%u] is not found.", exceptionInfo->streamid);
    }
}

void TaskExceptionHost::PrintUbDfxInfo(rtExceptionInfo_t *exceptionInfo, const Hccl::ErrorMessageReport &errorMessage)
{
    if (errorMessage.taskType == Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY ||
        errorMessage.taskType == Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY ||
        errorMessage.taskType == Hccl::TaskParamType::TASK_UB_INLINE_WRITE ||
        errorMessage.taskType == Hccl::TaskParamType::TASK_UB_REDUCE_INLINE ||
        errorMessage.taskType == Hccl::TaskParamType::TASK_UB) {
        HCCL_ERROR("errorMessage ubCqeStatus[%u], localEid[%s], remoteEid[%s]. ", (u32)errorMessage.ubCqeStatus,
            errorMessage.locEid.Describe().c_str(), errorMessage.rmtEid.Describe().c_str());
        auto reverseAddr = Hccl::IpAddress(errorMessage.locEid);
        auto addr = Hccl::IpAddress(reverseAddr.GetReverseEid());
        u32 devPhyId = Hccl::HrtGetDevicePhyIdByIndex(exceptionInfo->deviceid);
        auto rdmaHandle = Hccl::RdmaHandleManager::GetInstance().GetByIp(devPhyId, addr);
        PrintUbRegisters(static_cast<s32>(exceptionInfo->deviceid), rdmaHandle);
    }
}
} // namespace Hccl