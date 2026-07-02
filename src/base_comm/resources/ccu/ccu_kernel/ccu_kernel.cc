/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"
#include "ccu_rep_v1.h"
#include "ccu_kernel_resource.h"
#include "ccu_assist_v1.h"
#include "ccu_microcode_v1.h"

#include "ccu_types.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_dev_mgr_imp.h"
#include "env_config.h"
#include "ccu_rep_type_v1.h"

#include "hcomm_c_adpt.h"

#include "ccu_rep_context_v1.h"
#include "ccu_rep_funccall_v1.h"
#include "../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"

#include "ccu_log.h"

#include "hcom_common.h"

// todo: 引入头文件需要检查
#include "ccu_assist_v1.h"
#include "hccl_comm_pub.h"
#include "hcclCommDfx.h"
#include "task_info.h"
#include "task_param.h"

namespace hcomm {

constexpr uint32_t TOKEN_VALUE_INDEX = 2;
constexpr uint16_t INVALID_U16 = 65535;

template <typename T>
T CcuKernel::CreateResAssist(
    std::array<std::vector<T>, CCU_MAX_IODIE_NUM> &resRecord)
{
    constexpr uint32_t dieId = 0; // 先填个预留的dieId，后续分配
    resRecord[dieId].emplace_back(this);
    auto& item = resRecord[dieId].back();
    item.Reset(resRecord[dieId].size(), dieId);
    return item;
}

template <typename T>
std::vector<T> CcuKernel::CreateBlockResAssist(
    const uint32_t count,
    std::array<std::vector<T>, CCU_MAX_IODIE_NUM> &resRecord)
{
    std::vector<T> block;
    block.reserve(count);
    constexpr uint32_t dieId = 0; // 先填个预留的dieId，后续分配
    for (size_t i = 0; i < count; i++) {
        block.emplace_back(this);
        block.back().Reset(resRecord[dieId].size() + i, dieId);
    }
    resRecord[dieId].insert(resRecord[dieId].end(), block.begin(), block.end());
    return block;
}

CcuKernel::~CcuKernel()
{
}

static HcclResult GetDieIdByChannel(const ChannelHandle channel, uint32_t &dieId)
{
    void *channelPtr{nullptr};
    CHK_RET(static_cast<HcclResult>(HcommChannelGet(channel, &channelPtr)));
    auto *channelImpl = dynamic_cast<CcuUrmaChannel *>(static_cast<Channel *>(channelPtr));
    if (channelImpl == nullptr) {
        HCCL_ERROR("[%s] failed to cast channel[0x%llx] to CcuUrmaChannel", __func__, channel);
        return HcclResult::HCCL_E_PTR;
    }
    dieId = channelImpl->GetDieId();
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult GetDieIdByChannels(const std::unordered_set<ChannelHandle> &channels, uint32_t &dieId)
{
    if (channels.empty()) {
        int32_t devLogicId = HcclGetThreadDeviceId();
        for (uint32_t die = 0; die < CCU_MAX_IODIE_NUM; die++) {
            bool enableFlag = false;
            CHK_RET(static_cast<HcclResult>(CcuGetDieEnableInfo(devLogicId, die, enableFlag)));
            if (enableFlag) {
                dieId = die;
                return HcclResult::HCCL_SUCCESS;
            }
        }

        HCCL_ERROR("[CcuKernel][%s] failed, all dies are disable, devLogicId[%d].",
            __func__, devLogicId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t firstDieId = 0;
    CHK_RET(GetDieIdByChannel(*channels.begin(), firstDieId));
    for (const auto channel : channels) {
        uint32_t nextDieId = 0;
        CHK_RET(GetDieIdByChannel(channel, nextDieId));
        if (firstDieId != nextDieId) {
            HCCL_ERROR("[%s] failed, the dies of channels are not same.", __func__);
            return HcclResult::HCCL_E_PARA;
        }
    }

    dieId = firstDieId;
    return HcclResult::HCCL_SUCCESS;
}

static void MoveResourcesToDie(CcuRepResource &res, uint32_t targetDieId)
{
    if (targetDieId == 0) return; // 初始资源位于die0，不用设置
    
    auto moveAndSet = [&](auto &arr) {
        arr[targetDieId] = std::move(arr[0]);
        for (auto &item : arr[targetDieId]) item.SetDieId(targetDieId);
    };
    
    moveAndSet(res.ccubufs);
    moveAndSet(res.blockCcubufs);
    moveAndSet(res.executor);
    moveAndSet(res.blockExecutor);
    moveAndSet(res.completedEvent);
    moveAndSet(res.blockCompletedEvent);
    moveAndSet(res.address);
    moveAndSet(res.continuousVariable);
    moveAndSet(res.variable);
    moveAndSet(res.localNotify);
}

HcclResult CcuKernel::SetupProfilingInfo(const char *kernelFuncName)
{
    if (kernelFuncName == nullptr || strlen(kernelFuncName) == 0) {
        name_ = std::string("CCU_KERNEL"); // 默认名称
        AddSqeProfiling(name_);
        return HcclResult::HCCL_SUCCESS;
    }

    constexpr size_t MAX_KERNEL_FUNC_NAME_LEN = 128;
    const auto nameLen = strlen(kernelFuncName);
    if (nameLen > MAX_KERNEL_FUNC_NAME_LEN) {
        name_ = std::string(kernelFuncName, MAX_KERNEL_FUNC_NAME_LEN);
        HCCL_WARNING("[CcuKernel][%s] kernelFuncName is too long, reset to %s.",
            __func__, name_.c_str());
    }

    // 生成SQE粒度profiling信息，此时未选择die，默认die 0
    AddSqeProfiling(name_);
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult UpdateProfilingInfo(
    std::vector<CcuProfilingInfo> &profilingInfos, uint32_t dieId,
    const std::string &kernelName)
{
    if (dieId == 0) {
        // 与默认dieId相同，不需要修改
        return HcclResult::HCCL_SUCCESS;
    }

    // 正常情况仅首个info包含die信息，仅应为CCU_TASK_PROFILING类型
    if (UNLIKELY(profilingInfos.empty())) {
        // profiling不属于主流程，不打断算子业务
        HCCL_INFO("[%s] passed, profiling infos are empty, ccu kernel func[%s].",
            __func__, kernelName.c_str());
        return HcclResult::HCCL_SUCCESS;
    }

    // 根据选择的die跟新profiling信息
    for (auto &info : profilingInfos) {
        info.dieId = dieId;
    }

    HCCL_INFO("[%s] reset profiling info dieId to [%u], ccu kernel func[%s].",
        __func__, dieId, kernelName.c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernel::SelectDie()
{
    uint32_t dieId{0};
    CHK_RET(GetDieIdByChannels(channels_, dieId));
    CHK_PRT_RET(dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR("[CcuKernel][%s] failed, dieId[%u] should be less than [%u].",
            __func__, dieId, CCU_MAX_IODIE_NUM),
        HcclResult::HCCL_E_PARA);
    SetDieId(dieId);
    MoveResourcesToDie(res_, dieId);
    (void)UpdateProfilingInfo(profilingInfo, dieId, name_);

    return HcclResult::HCCL_SUCCESS;
}

CcuResult CcuKernel::ValidateTaskArgs(const uint64_t *taskArgs, uint32_t argsNum) const
{
    if (loadArgUsedSet_.size() != argsNum) {
        HCCL_ERROR("[CcuKernel][%s] failed, args number does not match the Load instruction, "
            "argsNum = %u, loaded = %zu", __func__, argsNum, loadArgUsedSet_.size());
        return CcuResult::CCU_E_INTERNAL;
    }
    for (uint32_t i = 0; i < argsNum; ++i) {
        if (loadArgUsedSet_.count(i) == 0) {
            HCCL_ERROR("[CcuKernel][%s] failed, argId %u not loaded (argsNum=%u)",
                __func__, i, argsNum);
            return CcuResult::CCU_E_INTERNAL;
        }
    }
    if (argsNum != 0) {
        CCU_CHK_PTR_NULL(taskArgs);
    }
    if (instrInfo_.missionInstrCount == 0 || instrInfo_.instrVec.empty()) {
        HCCL_ERROR("[CcuKernel][%s] failed, mission instructions are empty, "
            "the kernel is not been translated yet.", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

void CcuKernel::FillTaskParam(CcuTaskParam &param, uint32_t index, uint32_t seqNum,
    const uint64_t *taskArgs, uint32_t argsNum) const
{
    param.dieId       = GetDieId();
    param.missionId   = GetMissionId();
    param.instStartId = instrInfo_.missionStartInstrId + index * CCU_SQE_ARGS_LEN;
    param.key         = GetMissionKey();
    param.argSize     = CCU_SQE_ARGS_LEN;

    const uint32_t preMissionInsCnt = index * CCU_SQE_ARGS_LEN;
    const bool isLast = (index == seqNum - 1);
    param.instCnt = isLast ? (instrInfo_.missionInstrCount - preMissionInsCnt) : CCU_SQE_ARGS_LEN;

    if (argsNum > preMissionInsCnt) {
        const uint32_t argsToCopy = isLast
            ? std::min(argsNum - preMissionInsCnt, CCU_SQE_ARGS_LEN)
            : CCU_SQE_ARGS_LEN;
        std::copy(taskArgs + preMissionInsCnt, taskArgs + preMissionInsCnt + argsToCopy,
                  std::begin(param.args));
    }

    HCCL_INFO("[GeneTaskParam]task Param, dieId[%u] missionId[%u] instStartId[%u] instCnt[%u], argSize[%u]",
              param.dieId, param.missionId, param.instStartId, param.instCnt, param.argSize);
}

CcuResult CcuKernel::GeneTaskParams(const uint64_t *taskArgs, uint32_t argsNum,
    std::vector<CcuTaskParam> &taskParams)
{
    CCU_CHK_RET(ValidateTaskArgs(taskArgs, argsNum));

    // 如果agrs数量超过sqe arg的最大数量，则返回多个TaskParam，前面几个只从sqe中加载args;
    // args数量大于等于0、小于等于最大值时，返回1个TaskParam
    const uint32_t seqNum
        = (argsNum / CCU_SQE_ARGS_LEN) + ((argsNum % CCU_SQE_ARGS_LEN) == 0 ? 0 : 1) + (argsNum == 0 ? 1 : 0);

    const uint32_t preMissonSqeInsCnt = (seqNum - 1) * CCU_SQE_ARGS_LEN;
    if (instrInfo_.missionInstrCount < preMissonSqeInsCnt) {
        HCCL_ERROR("[CcuKernel][%s] failed, missionInstrCount[%u] should be greater "
            "than preMissonSqeInsCnt[%u].", __func__, instrInfo_.missionInstrCount,
            preMissonSqeInsCnt);
        return CcuResult::CCU_E_INTERNAL;
    }

    taskParams.resize(seqNum);
    for (uint32_t index = 0; index < seqNum; index++) {
        FillTaskParam(taskParams[index], index, seqNum, taskArgs, argsNum);
    }

    return CcuResult::CCU_SUCCESS;
}

HcclResult CcuKernel::CreateVariable(const ChannelHandle channel, uint32_t varIndex, CcuRep::Variable *var)
{
    channels_.insert(channel);

    void *channelPtr{nullptr};
    CHK_RET(static_cast<HcclResult>(HcommChannelGet(channel, &channelPtr)));
    auto *channelImpl = dynamic_cast<CcuUrmaChannel *>(static_cast<Channel *>(channelPtr));
    if (channelImpl == nullptr) {
        HCCL_ERROR("[%s] failed to cast channel[0x%llx] to CcuUrmaChannel", __func__, channel);
        return HcclResult::HCCL_E_PTR;
    }
    uint32_t locXnId{0};
    CHK_RET(channelImpl->GetLocXnByIndex(varIndex, locXnId));
    var->Reset(locXnId, channelImpl->GetDieId());
    return HcclResult::HCCL_SUCCESS;
}

CcuRepResource &CcuKernel::GetResource()
{
    return res_;
}

CcuResReq CcuKernel::GetResourceRequest()
{
    CcuResReq req;
    uint32_t dieId = GetDieId();
    req.msReq[dieId]              = res_.ccubufs[dieId].size();
    req.blockMsReq[dieId]         = res_.blockCcubufs[dieId].size();
    req.ckeReq[dieId]             = res_.completedEvent[dieId].size()
                                    + res_.localNotify[dieId].size();
    req.blockCkeReq[dieId]        = res_.blockCompletedEvent[dieId].size();
    req.loopEngineReq[dieId]      = res_.executor[dieId].size();
    req.blockLoopEngineReq[dieId] = res_.blockExecutor[dieId].size();
    req.gsaReq[dieId]             = res_.address[dieId].size();
    req.xnReq[dieId]              = res_.variable[dieId].size();
    req.continuousXnReq[dieId]    = res_.continuousVariable[dieId].size();

    req.missionReq.reqType           = MissionReqType::FUSION_MULTIPLE_DIE;
    req.missionReq.req[dieId] = 1;

    auto info
        = Hccl::StringFormat("resource request: dieId[%u], ms[%u], blockMs[%u], cke[%u], blockCke[%u], "
                       "loopEngine[%u], blockLoopEngine[%u], gsa[%u], xn[%u], continuous xn[%u], missionId[%u]",
                       dieId, req.msReq[dieId], req.blockMsReq[dieId], req.ckeReq[dieId], req.blockCkeReq[dieId],
                       req.loopEngineReq[dieId], req.blockLoopEngineReq[dieId], req.gsaReq[dieId], req.xnReq[dieId],
                       req.continuousXnReq[dieId], req.missionReq.req[dieId]);

    HCCL_INFO("%s", info.c_str());

    return req;
}

template<typename HandleType, typename ResourceType>
static CcuResult GetResourceByHandle(
    std::unordered_map<HandleType, ResourceType> &resourceMap, 
    HandleType handle, ResourceType **resource, const char *resourceType)
{
    auto iter = resourceMap.find(handle);
    if (iter == resourceMap.end()) {
        HCCL_ERROR("[%s] failed to find %s by handle: 0x%llx", __func__, resourceType, handle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    // ccu资源本身可能重载=，对象赋值会被转换成指令，导致流程失败
    *resource = &(iter->second);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::GetVariableByHandle(CcuVariableHandle varHandle, CcuRep::Variable **variable)
{
    return GetResourceByHandle(ccuVarMap_, varHandle, variable, "variable");
}
//Alloc 相关接口
CcuResult CcuKernel::VariableAlloc(CcuVariableHandle *varHandle)
{
    const auto &var = CreateResAssist(res_.continuousVariable);
    CcuVariableHandle handle = ccuVarMap_.size();
    ccuVarMap_.emplace(handle, var);

    *varHandle = handle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::AddressAlloc(CcuAddressHandle *addrHandle)
{
    const auto &addr = CreateResAssist(res_.address);
    CcuAddressHandle handle = ccuAddrMap_.size();
    ccuAddrMap_.emplace(handle, addr);
    *addrHandle = handle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::EventAlloc(CcuEventHandle *eventHandle)
{
    const auto &event = CreateResAssist(res_.completedEvent);
    CcuEventHandle handle = ccuEventMap_.size();
    ccuEventMap_.emplace(handle, event);
    *eventHandle = handle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::BufferAlloc(CcuBufferHandle *bufHandle)
{
    const auto &buf = CreateResAssist(res_.ccubufs);
    CcuBufferHandle handle = ccuBufferMap_.size();
    ccuBufferMap_.emplace(handle, buf);
    *bufHandle = handle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::LocalAddrAlloc(CcuLocalAddrHandle *localAddrHandle, CcuAddressHandle *addrHandle, CcuVariableHandle *tokenHandle)
{
    auto localAddr = CreateLocalAddr();
    
    CcuAddressHandle aHandle = ccuAddrMap_.size();
    ccuAddrMap_.emplace(aHandle, localAddr.addr);

    CcuVariableHandle tHandle = ccuVarMap_.size();
    ccuVarMap_.emplace(tHandle, localAddr.token);

    CcuLocalAddrHandle laHandle = ccuLocalAddrMap_.size();
    ccuLocalAddrMap_.emplace(laHandle, localAddr);

    *localAddrHandle = laHandle;
    *addrHandle = aHandle;
    *tokenHandle = tHandle;
    return CcuResult::CCU_SUCCESS;
    
}
CcuResult CcuKernel::RemoteAddrAlloc(CcuRemoteAddrHandle *remoteAddrHandle, CcuAddressHandle *addrHandle, CcuVariableHandle *tokenHandle)
{
    auto remoteAddr = CreateRemoteAddr();

    CcuAddressHandle aHandle = ccuAddrMap_.size();
    ccuAddrMap_.emplace(aHandle, remoteAddr.addr);

    CcuVariableHandle tHandle = ccuVarMap_.size();
    ccuVarMap_.emplace(tHandle, remoteAddr.token);

    CcuRemoteAddrHandle raHandle = ccuRemoteAddrMap_.size();
    ccuRemoteAddrMap_.emplace(raHandle, remoteAddr);

    *remoteAddrHandle = raHandle;
    *addrHandle = aHandle;
    *tokenHandle = tHandle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::BlockVariableAlloc(CcuVariableHandle *varHandles, uint32_t count)
{
    const auto& var = CreateBlockResAssist(count, res_.continuousVariable);
    for (uint32_t i = 0; i < count; i++) {  
        CcuVariableHandle handle = ccuVarMap_.size();   
        ccuVarMap_.emplace(handle, var[i]);
        varHandles[i] = handle;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::BlockEventAlloc(CcuEventHandle *eventHandles, uint32_t count)
{
    const auto& event = CreateBlockResAssist(count, res_.blockCompletedEvent);
    for (uint32_t i = 0; i < count; i++) {  
        CcuEventHandle handle = ccuEventMap_.size();
        ccuEventMap_.emplace(handle, event[i]);
        eventHandles[i] = handle;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::BlockBufferAlloc(CcuBufferHandle *bufHandles, uint32_t count)
{
    const auto& buffer = CreateBlockResAssist(count, res_.blockCcubufs);
    for (uint32_t i = 0; i < count; i++) {  
        CcuBufferHandle handle = ccuBufferMap_.size();
        ccuBufferMap_.emplace(handle, buffer[i]);
        bufHandles[i] = handle;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableCreateByChannel(ChannelHandle channel, uint32_t varIndex, CcuVariableHandle *varHandle)
{
    CcuRep::Variable var(this);
    CCU_CHK_RET(CreateVariable(channel, varIndex, &var));
    CcuVariableHandle handle = ccuVarMap_.size();
    ccuVarMap_.emplace(handle, var);
    *varHandle = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableAssignImm(CcuVariableHandle varHandle, uint64_t immediate)
{
    CcuRep::Variable *variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));
    // 通过符号重载实现，内部记录rep；异常由入口 HcommCcuKernelRegister 的
    // CCU_EXCEPTION_HANDLE_BEGIN/END 统一接住，无需在此局部 try/catch。
    (*variable) = immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableAssignVar(CcuVariableHandle varHandle, CcuVariableHandle varA)
{
    CcuRep::Variable *variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));
    CcuRep::Variable *variableA{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varA, &variableA));
    // 通过符号重载实现，内部记录rep；异常由入口统一 catch。
    (*variable) = (*variableA);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableAddVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varA, CcuVariableHandle varB)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varA, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varB, &rightVar));

    // 通过符号重载实现，内部记录rep；异常由入口统一 catch。
    *resVar = *leftVar + *rightVar;
    return CcuResult::CCU_SUCCESS;
}

/*========== Event信号同步类 相关接口 ==========*/
CcuResult CcuKernel::EventRecord(CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    // 复用已有的 RecordEvent 实现（内部 Append CcuRepLocRecordEvent）
    CCU_CHK_RET(RecordEvent(*event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::EventWait(CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    // 复用已有的 WaitEvent 实现（内部 Append CcuRepLocWaitEvent）
    CCU_CHK_RET(WaitEvent(*event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LocalNotifyRecord(const char *notifyTag, const uint32_t mask)
{
    if (notifyTag == nullptr) {
        HCCL_ERROR("[CcuKernel][%s] notifyTag is nullptr, please check.", __func__);
        return CcuResult::CCU_E_PTR;
    }
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel][%s] is not supported in loop block, please check.", __func__);
        return CcuResult::CCU_E_NOT_SUPPORT;
    }

    const std::string tagKey(notifyTag);

    auto &sharedNotifies = importedRes_.sharedNotifies;
    if (sharedNotifies.find(tagKey) == sharedNotifies.end()) {
        CcuRep::LocalNotify localNotify;
        sharedNotifies.insert({tagKey, localNotify});
    }

    Append(std::make_shared<CcuRep::CcuRepRecordSharedNotify>(sharedNotifies.at(tagKey), mask));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LocalNotifyWait(const char *notifyTag, const uint32_t mask)
{
    if (notifyTag == nullptr) {
        HCCL_ERROR("[CcuKernel][%s] notifyTag is nullptr, please check.", __func__);
        return CcuResult::CCU_E_PTR;
    }

    const std::string tagKey(notifyTag);

    auto &sharedNotifies = exportedRes_.sharedNotifies;
    if (sharedNotifies.find(tagKey) == sharedNotifies.end()) {
        CcuRep::LocalNotify notify = CreateLocalNotify();
        exportedRes_.sharedNotifies.insert({tagKey, notify});
    }

    bool isProfiling = CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    Append(std::make_shared<CcuRep::CcuRepLocWaitNotify>(
        exportedRes_.sharedNotifies.at(tagKey), mask, isProfiling));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::NotifyRecord(const ChannelHandle channel,
    uint32_t remoteNotifyIdx, uint32_t mask)
{
    channels_.insert(channel);
    Append(std::make_shared<CcuRep::CcuRepRemPostSem>(channel, remoteNotifyIdx, mask));
    return CCU_SUCCESS;
}

CcuResult CcuKernel::NotifyWait(const ChannelHandle channel, uint32_t localNotifyIdx, uint32_t mask)
{
    bool isProfiling = CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    if (isProfiling) {
        CCU_CHK_RET(static_cast<HcclResult>(AddProfiling(channel, "NotifyWait", localNotifyIdx, mask)));
    }
    Append(std::make_shared<CcuRep::CcuRepRemWaitSem>(channel, localNotifyIdx, mask, isProfiling));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::WriteVariableWithNotify(const ChannelHandle channel, CcuVariableHandle varHandle,
    uint32_t remoteVarIdx, uint32_t remoteNotifyIdx, uint32_t mask)
{
    channels_.insert(channel);
    CcuRep::Variable *var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    Append(std::make_shared<CcuRep::CcuRepRemPostVar>(*var, channel, remoteVarIdx, remoteNotifyIdx, mask));
    return CcuResult::CCU_SUCCESS;
}


//加载类 相关接口
CcuResult  CcuKernel::LoadArg(CcuVariableHandle varHandle, uint32_t argId)
{
    loadArgUsedSet_.insert(argId);
    CcuRep::Variable *var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle,&var));
    auto loadArgRep = std::make_shared<CcuRep::CcuRepLoadArg>(
        *var, argId % CCU_SQE_ARGS_LEN, static_cast<uint16_t>(argId));
    Append(loadArgRep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::CheckContinuousVariables(CcuVariableHandle varHandle, uint32_t num,
    const CcuRep::Variable &baseVar, const char *tag)
{
    if (num <= 1) {
        return CcuResult::CCU_SUCCESS;
    }
    for (uint32_t i = 1; i < num; i++) {
        CcuRep::Variable *nextVar{nullptr};
        CCU_CHK_RET(GetVariableByHandle(varHandle + i, &nextVar));
        if (nextVar->Id() != baseVar.Id() + i) {
            HCCL_ERROR("[CcuKernel][%s] variables not continuous at index %u, "
                       "expected Id %u but got %u", tag, i, baseVar.Id() + i, nextVar->Id());
            return HCCL_TO_CCU_RET(HCCL_E_PARA);
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoadVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    CcuRep::Variable *var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(varHandle, num, *var, "LoadVariable"));
    Append(std::make_shared<CcuRep::CcuRepLoad>(addr, *var, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::CcuLoadVarFromVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    CcuRep::Variable *addrVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(addrHandle, &addrVar));
    CcuRep::Variable *var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(varHandle, num, *var, "LoadVar dst"));
    Append(std::make_shared<CcuRep::CcuRepLoadVar>(*addrVar, *var, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::StoreVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    CcuRep::Variable *var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(varHandle, num, *var, "StoreVariable"));
    Append(std::make_shared<CcuRep::CcuRepStore>(*var, addr, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::CcuStoreVarToVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    CcuRep::Variable *addrVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(addrHandle, &addrVar));
    CcuRep::Variable *var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(varHandle, num, *var, "StoreVar src"));
    Append(std::make_shared<CcuRep::CcuRepStoreVar>(*var, *addrVar, num));
    return CcuResult::CCU_SUCCESS;
}

//本地数据拷贝 相关实现
CcuResult CcuKernel::LocalCopyMemToBuffer(CcuBufferHandle dstHandle, CcuLocalAddrHandle srcHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::CcuBuf *dst{nullptr};
    CCU_CHK_RET(GetBufferByHandle(dstHandle, &dst));
    CcuRep::LocalAddr *src{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(srcHandle, &src));
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = LocalCopyNb(*dst, *src, *len, *event, mask);  // 复用 protected
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::LocalCopyBufferToMem(CcuLocalAddrHandle dstHandle, CcuBufferHandle srcHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::LocalAddr *dst{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(dstHandle, &dst));
    CcuRep::CcuBuf *src{nullptr};
    CCU_CHK_RET(GetBufferByHandle(srcHandle, &src));
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = LocalCopyNb(*dst, *src, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::LocalCopyMemToMem(CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::LocalAddr *dst{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(dstHandle, &dst));
    CcuRep::LocalAddr *src{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(srcHandle, &src));
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = LocalCopyNb(*dst, *src, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

//本地reduce 相关实现
CcuResult CcuKernel::LocalMemReduce(CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle,
    CcuVariableHandle lenHandle, HcclDataType dataType,
    HcclReduceOp opType, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::LocalAddr *dst{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(dstHandle, &dst));
    CcuRep::LocalAddr *src{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(srcHandle, &src));
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = LocalReduceNb(*dst, *src, *len, dataType, opType, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::LocalBufferReduce(CcuBufferHandle* bufHandles, uint32_t count,
    HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType, CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    std::vector<CcuRep::CcuBuf> bufs(count);
    for (uint32_t i = 0; i < count; i++) {
        CcuRep::CcuBuf *buf{nullptr};
        CCU_CHK_RET(GetBufferByHandle(bufHandles[i], &buf));
        bufs[i] = *buf;
    }
    auto ret = LocalReduceNb(bufs.data(), count, dataType, outputDataType, opType, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

/*========== 远端数据传输操作 ==========*/

CcuResult CcuKernel::ResolveBufRemoteLenEvent(CcuBufferHandle bufHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle,
    CcuRep::CcuBuf **buf, CcuRep::RemoteAddr **remote,
    CcuRep::Variable **len, CcuRep::CompletedEvent **event)
{
    CCU_CHK_RET(GetBufferByHandle(bufHandle, buf));
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, remote));
    CCU_CHK_RET(GetVariableByHandle(lenHandle, len));
    CCU_CHK_RET(GetEventByHandle(eventHandle, event));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::ReadMemToMem(ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::LocalAddr *local{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(localHandle, &local));
    CcuRep::RemoteAddr *remote{nullptr};
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, &remote));
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = ReadNb(channel, *local, *remote, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::ReadMemToBuffer(ChannelHandle channel, CcuBufferHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::CcuBuf *local{nullptr};
    CcuRep::RemoteAddr *remote{nullptr};
    CcuRep::Variable *len{nullptr};
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(ResolveBufRemoteLenEvent(localHandle, remoteHandle, lenHandle, eventHandle,
        &local, &remote, &len, &event));
    auto ret = ReadNb(channel, *local, *remote, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::ReadMemToMemReduce(ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle lenHandle, HcclDataType dataType,
    HcclReduceOp opType, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::LocalAddr *local{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(localHandle, &local));
    CcuRep::RemoteAddr *remote{nullptr};
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, &remote));
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = ReadReduceNb(channel, *local, *remote, *len, dataType, opType, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::WriteMemToMem(ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::RemoteAddr *remote{nullptr};
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, &remote));
    CcuRep::LocalAddr *local{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(localHandle, &local));
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = WriteNb(channel, *remote, *local, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::WriteBufferToMem(ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuBufferHandle localHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::CcuBuf *local{nullptr};
    CcuRep::RemoteAddr *remote{nullptr};
    CcuRep::Variable *len{nullptr};
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(ResolveBufRemoteLenEvent(localHandle, remoteHandle, lenHandle, eventHandle,
        &local, &remote, &len, &event));
    auto ret = WriteNb(channel, *remote, *local, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::WriteMemToMemReduce(ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
    CcuVariableHandle lenHandle, HcclDataType dataType,
    HcclReduceOp opType, CcuEventHandle eventHandle, uint32_t mask)
{
    CcuRep::RemoteAddr *remote{nullptr};
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, &remote));
    CcuRep::LocalAddr *local{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(localHandle, &local));
    CcuRep::Variable *len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent *event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = WriteReduceNb(channel, *remote, *local, *len, dataType, opType, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

void CcuKernel::FlushClosablePendingIfs()
{
    if (isFlushing_) {
        return;
    }
    isFlushing_ = true;
    while (IfLabelStackTopIsClosable()) {
        const char *lbl = IfLabelStackPop();
        if (lbl != nullptr) {
            IfEnd(lbl);
        }
    }
    isFlushing_ = false;
}

void CcuKernel::Append(std::shared_ptr<CcuRep::CcuRepBase> rep)
{
    FlushClosablePendingIfs();
    CcuRep::CcuRepContext::Append(rep);
}

CcuResult CcuKernel::IfBegin(CcuVariableHandle varHandle, uint64_t immediate,
    CcuConditionType condType, const char *label)
{
    if (loopBodyDepth_ > 0) {
        HCCL_ERROR("[%s] CCU_IF is not allowed inside a ccu::Loop body (label='%s')",
            __func__, label != nullptr ? label : "(null)");
        return CcuResult::CCU_E_INTERNAL;
    }
    CcuRep::Variable *variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    FlushClosablePendingIfs();

    std::string labelStr(label);
    if (pendingIfCtx_.find(labelStr) != pendingIfCtx_.end()) {
        HCCL_ERROR("[%s] label '%s' already has a pending IfBegin without IfEnd", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    std::string elseLabelStr = labelStr + "_else";
    std::string endLabelStr = labelStr + "_end";

    auto elseLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(elseLabelStr);
    auto endLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(endLabelStr);
    auto targetVar = CcuRep::CreateVariable(this);

    // Invert condition: "if EQ, execute block" means "jump over block when NE"
    std::shared_ptr<CcuRep::CcuRepJumpBase> jump{nullptr};
    if (condType == CCU_CONDITION_EQ) {
        jump = std::make_shared<CcuRep::CcuRepJumpNE>(
            elseLabelStr, targetVar, *variable, immediate);
    } else if (condType == CCU_CONDITION_NE) {
        jump = std::make_shared<CcuRep::CcuRepJumpEQ>(
            elseLabelStr, targetVar, *variable, immediate);
    } else {
        HCCL_ERROR("[%s] unsupported condition type: %d", __func__, condType);
        return CcuResult::CCU_E_PARA;
    }

    jump->Reference(elseLabel);
    Append(jump);

    PendingIfContext ctx;
    ctx.elseLabel = elseLabel;
    ctx.endLabel = endLabel;
    ctx.hasElse = false;
    pendingIfCtx_.emplace(labelStr, std::move(ctx));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::IfElse(const char *label)
{
    if (loopBodyDepth_ > 0) {
        HCCL_ERROR("[%s] CCU_ELSE is not allowed inside a ccu::Loop body (label='%s')",
            __func__, label != nullptr ? label : "(null)");
        return CcuResult::CCU_E_INTERNAL;
    }

    std::string labelStr(label);
    auto iter = pendingIfCtx_.find(labelStr);
    if (iter == pendingIfCtx_.end()) {
        HCCL_ERROR("[%s] no matching IfBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    if (iter->second.hasElse) {
        HCCL_ERROR("[%s] label '%s' already has an IfElse", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    // At end of then-block: unconditional jump past else-block to endLabel
    std::string endLabelStr = labelStr + "_end";
    auto skipElseVar = CcuRep::CreateVariable(this);
    auto skipElseJump = std::make_shared<CcuRep::CcuRepJump>(
        endLabelStr, skipElseVar);
    skipElseJump->Reference(iter->second.endLabel);
    Append(skipElseJump);

    // Place the else label (entry point of else-block)
    Append(iter->second.elseLabel);

    iter->second.hasElse = true;

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::IfEnd(const char *label)
{
    std::string labelStr(label);
    auto iter = pendingIfCtx_.find(labelStr);
    if (iter == pendingIfCtx_.end()) {
        HCCL_ERROR("[%s] no matching IfBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    if (iter->second.hasElse) {
        // Had else-block: place endLabel after else-block
        Append(iter->second.endLabel);
    } else {
        // No else-block: place elseLabel as the skip target
        Append(iter->second.elseLabel);
    }

    pendingIfCtx_.erase(iter);

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::WhileBegin(CcuVariableHandle varHandle, uint64_t immediate,
    CcuConditionType condType, const char *label)
{
    if (loopBodyDepth_ > 0) {
        HCCL_ERROR("[%s] CCU_WHILE is not allowed inside a ccu::Loop body (label='%s')",
            __func__, label != nullptr ? label : "(null)");
        return CcuResult::CCU_E_INTERNAL;
    }

    CcuRep::Variable *variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    std::string labelStr(label);
    if (pendingWhileCtx_.find(labelStr) != pendingWhileCtx_.end()) {
        HCCL_ERROR("[%s] label '%s' already has a pending WhileBegin without WhileEnd", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    std::string beginLabelStr = labelStr + "_begin";
    std::string endLabelStr = labelStr + "_end";

    auto beginLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(beginLabelStr);
    auto endLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(endLabelStr);

    Append(beginLabel);

    auto targetVar = CcuRep::CreateVariable(this);
    std::shared_ptr<CcuRep::CcuRepJumpBase> jump{nullptr};
    if (condType == CCU_CONDITION_EQ) {
        jump = std::make_shared<CcuRep::CcuRepJumpNE>(
            endLabelStr, targetVar, *variable, immediate);
    } else if (condType == CCU_CONDITION_NE) {
        jump = std::make_shared<CcuRep::CcuRepJumpEQ>(
            endLabelStr, targetVar, *variable, immediate);
    } else {
        HCCL_ERROR("[%s] unsupported condition type: %d", __func__, condType);
        return CcuResult::CCU_E_PARA;
    }

    jump->Reference(endLabel);
    Append(jump);

    PendingWhileContext ctx;
    ctx.beginLabel = beginLabel;
    ctx.endLabel = endLabel;
    ctx.varHandle = varHandle;
    ctx.immediate = immediate;
    ctx.condType = condType;
    pendingWhileCtx_.emplace(labelStr, std::move(ctx));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::WhileEnd(const char *label)
{
    std::string labelStr(label);
    auto iter = pendingWhileCtx_.find(labelStr);
    if (iter == pendingWhileCtx_.end()) {
        HCCL_ERROR("[%s] no matching WhileBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    std::string beginLabelStr = labelStr + "_begin";
    auto loopBackVar = CcuRep::CreateVariable(this);
    auto loopBackJump = std::make_shared<CcuRep::CcuRepJump>(
        beginLabelStr, loopBackVar);
    loopBackJump->Reference(iter->second.beginLabel);
    Append(loopBackJump);

    Append(iter->second.endLabel);

    pendingWhileCtx_.erase(iter);

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::DoWhileBegin(const char *label)
{
    if (loopBodyDepth_ > 0) {
        HCCL_ERROR("[%s] CCU_DO is not allowed inside a ccu::Loop body (label='%s')",
            __func__, label != nullptr ? label : "(null)");
        return CcuResult::CCU_E_INTERNAL;
    }

    std::string labelStr(label);
    if (pendingDoWhileCtx_.find(labelStr) != pendingDoWhileCtx_.end()) {
        HCCL_ERROR("[%s] label '%s' already has a pending DoWhileBegin without DoWhileEnd", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    std::string beginLabelStr = labelStr + "_begin";
    auto beginLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(beginLabelStr);
    Append(beginLabel);

    PendingDoWhileContext ctx;
    ctx.beginLabel = beginLabel;
    pendingDoWhileCtx_.emplace(labelStr, std::move(ctx));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::DoWhileEnd(CcuVariableHandle varHandle, uint64_t immediate,
    CcuConditionType condType, const char *label)
{
    CcuRep::Variable *variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    std::string labelStr(label);
    auto iter = pendingDoWhileCtx_.find(labelStr);
    if (iter == pendingDoWhileCtx_.end()) {
        HCCL_ERROR("[%s] no matching DoWhileBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    std::string beginLabelStr = labelStr + "_begin";
    auto targetVar = CcuRep::CreateVariable(this);
    std::shared_ptr<CcuRep::CcuRepJumpBase> jump{nullptr};

    // "condition true => continue looping" means jump back to begin when condition holds
    if (condType == CCU_CONDITION_EQ) {
        jump = std::make_shared<CcuRep::CcuRepJumpEQ>(
            beginLabelStr, targetVar, *variable, immediate);
    } else if (condType == CCU_CONDITION_NE) {
        jump = std::make_shared<CcuRep::CcuRepJumpNE>(
            beginLabelStr, targetVar, *variable, immediate);
    } else {
        HCCL_ERROR("[%s] unsupported condition type: %d", __func__, condType);
        return CcuResult::CCU_E_PARA;
    }

    jump->Reference(iter->second.beginLabel);
    Append(jump);

    pendingDoWhileCtx_.erase(iter);

    return CcuResult::CCU_SUCCESS;
}

// 控制流标签栈实体
void CcuKernel::IfLabelStackPush(const char *label)
{
    iflabelStack_.push_back({label, false});
}
void CcuKernel::IfLabelStackMarkBodyDone()
{
    if (iflabelStack_.empty()) {
        HCCL_ERROR("[CcuKernel::IfLabelStack][MarkBodyDone] stack is empty");
        return;
    }
    iflabelStack_.back().bodyDone = true;
}
const char *CcuKernel::IfLabelStackPopForElse()
{
    if (iflabelStack_.empty()) {
        HCCL_ERROR("[CcuKernel::IfLabelStack][PopForElse] orphan CCU_ELSE: "
                   "no matching CCU_IF on the stack");
        return nullptr;
    }
    if (!iflabelStack_.back().bodyDone) {
        HCCL_ERROR("[CcuKernel::IfLabelStack][PopForElse] CCU_ELSE called while "
                   "top if-body is still InBody (label='%s')",
                   iflabelStack_.back().label != nullptr
                       ? iflabelStack_.back().label : "(null)");
        return nullptr;
    }
    const char *label = iflabelStack_.back().label;
    iflabelStack_.pop_back();
    return label;
}
bool CcuKernel::IfLabelStackTopIsClosable()
{
    return !iflabelStack_.empty() && iflabelStack_.back().bodyDone;
}

const char *CcuKernel::IfLabelStackPop()
{
    if (iflabelStack_.empty()) {
        return nullptr;
    }
    const char *label = iflabelStack_.back().label;
    iflabelStack_.pop_back();
    return label;
}

void CcuKernel::DoWhileLabelStackPush(const char *label)
{
    doWhileLabelStack_.push_back(label);
}

const char *CcuKernel::DoWhileLabelStackPopForWhile()
{
    if (loopBodyDepth_ > 0) {
        return nullptr;
    }
    if (doWhileLabelStack_.empty()) {
        return nullptr;
    }
    const char *label = doWhileLabelStack_.back();
    doWhileLabelStack_.pop_back();
    return label;
}

CcuResult CcuKernel::GetAddressByHandle(CcuAddressHandle addrHandle, CcuRep::Address **address)
{
    return GetResourceByHandle(ccuAddrMap_, addrHandle, address, "address");
}

// addr = 立即数 → CcuRepAssign(Address, uint64_t)
CcuResult CcuKernel::AddressAssignImm(CcuAddressHandle addrHandle, uint64_t immediate)
{
    CcuRep::Address *address{nullptr};
    CCU_CHK_RET(GetAddressByHandle(addrHandle, &address));
    (*address) = immediate;
    return CcuResult::CCU_SUCCESS;
}

// addr = variable → CcuRepAssign(Address, Variable)
CcuResult CcuKernel::AddressAssignVar(CcuAddressHandle addrHandle, CcuVariableHandle varHandle)
{
    CcuRep::Address *address{nullptr};
    CCU_CHK_RET(GetAddressByHandle(addrHandle, &address));

    CcuRep::Variable *variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    (*address) = (*variable);
    return CcuResult::CCU_SUCCESS;
}
// addr = addr → CcuRepAssign(Address, Address)
CcuResult CcuKernel::AddressAssignAddr(CcuAddressHandle dstAddrHandle, CcuAddressHandle srcAddrHandle)
{
    CcuRep::Address *dstAddress{nullptr};
    CCU_CHK_RET(GetAddressByHandle(dstAddrHandle, &dstAddress));

    CcuRep::Address *srcAddress{nullptr};
    CCU_CHK_RET(GetAddressByHandle(srcAddrHandle, &srcAddress));

    (*dstAddress) = (*srcAddress);
    return CcuResult::CCU_SUCCESS;
}

// resAddr = lhsAddr + rhsVar → CcuRepAdd(Address, Address, Variable)
CcuResult CcuKernel::AddressAddVarToAddr(
    CcuAddressHandle resAddrHandle, CcuAddressHandle lhsAddrHandle, CcuVariableHandle rhsVarHandle)
{
    CcuRep::Address *resAddr{nullptr}, *lhsAddr{nullptr};
    CCU_CHK_RET(GetAddressByHandle(resAddrHandle, &resAddr));
    CCU_CHK_RET(GetAddressByHandle(lhsAddrHandle, &lhsAddr));

    CcuRep::Variable *rhsVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(rhsVarHandle, &rhsVar));

    *resAddr = *lhsAddr + *rhsVar;
    return CcuResult::CCU_SUCCESS;
}

// resAddr = addrA + addrB → CcuRepAdd(Address, Address, Address)
CcuResult CcuKernel::AddressAddAddrToAddr(
    CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, CcuAddressHandle addrBHandle)
{
    CcuRep::Address *resAddr{nullptr}, *addrA{nullptr}, *addrB{nullptr};
    CCU_CHK_RET(GetAddressByHandle(resAddrHandle, &resAddr));
    CCU_CHK_RET(GetAddressByHandle(addrAHandle, &addrA));
    CCU_CHK_RET(GetAddressByHandle(addrBHandle, &addrB));

    *resAddr = *addrA + *addrB;
    return CcuResult::CCU_SUCCESS;
}

// addr += variable → CcuRepAdd(Address, Variable) 就地加
CcuResult CcuKernel::AddressAddAssignVar(CcuAddressHandle addrHandle, CcuVariableHandle varHandle)
{
    CcuRep::Address *address{nullptr};
    CCU_CHK_RET(GetAddressByHandle(addrHandle, &address));

    CcuRep::Variable *variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    (*address) += (*variable);
    return CcuResult::CCU_SUCCESS;
}

// addr += addr → 等价于 addr = addr + otherAddr
CcuResult CcuKernel::AddressAddAssignAddr(CcuAddressHandle addrHandle, CcuAddressHandle otherHandle)
{
    CcuRep::Address *address{nullptr};
    CCU_CHK_RET(GetAddressByHandle(addrHandle, &address));

    CcuRep::Address *other{nullptr};
    CCU_CHK_RET(GetAddressByHandle(otherHandle, &other));

    (*address) = (*address) + (*other);
    return CcuResult::CCU_SUCCESS;
}

void CcuKernel::Load(const CcuRep::Variable &var)
{
    auto loadArgRep = std::make_shared<CcuRep::CcuRepLoadArg>(
        var, loadArgIndex_ % CCU_SQE_ARGS_LEN, static_cast<uint16_t>(loadArgIndex_));
    GetLGProfilingInfo().loadRep2ArgIdxMap[loadArgRep] = loadArgIndex_;
    Append(loadArgRep);
    loadArgIndex_++;
}

void CcuKernel::StoreVariable(const CcuRep::Variable &var, uint64_t addr)
{
    Append(std::make_shared<CcuRep::CcuRepStore>(var, addr));
}

void CcuKernel::LoadVariable(const CcuRep::Variable &src, const CcuRep::Variable &var)
{
    Append(std::make_shared<CcuRep::CcuRepLoadVar>(src, var));
}

HcclResult CcuKernel::RecordEvent(CcuRep::CompletedEvent event, uint32_t mask)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel][%s] is not supported in loop block, please check.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    Append(std::make_shared<CcuRep::CcuRepLocRecordEvent>(event, mask));
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::WaitEvent(CcuRep::CompletedEvent event, uint32_t mask)
{
    bool isProfiling = CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    auto rep = std::make_shared<CcuRep::CcuRepLocWaitEvent>(event, mask, isProfiling);
    if (isProfiling) {
        CHK_RET(static_cast<HcclResult>(AddProfiling("WaitEvent", rep->GetMask())));
    }
 	Append(rep);
    return HCCL_SUCCESS;
}

CcuResult CcuKernel::GetEventByHandle(CcuEventHandle eventHandle, CcuRep::CompletedEvent **event)
{
    return GetResourceByHandle(ccuEventMap_, eventHandle, event, "completedEvent");
}

/*
LocalAddr / RemoteAddr 相关接口
*/
CcuResult CcuKernel::GetLocalAddrByHandle(CcuLocalAddrHandle handle, CcuRep::LocalAddr **localAddr)
{
    return GetResourceByHandle(ccuLocalAddrMap_, handle, localAddr, "localAddr");
}

CcuResult CcuKernel::GetRemoteAddrByHandle(CcuRemoteAddrHandle handle, CcuRep::RemoteAddr **remoteAddr)
{
    return GetResourceByHandle(ccuRemoteAddrMap_, handle, remoteAddr, "remoteAddr");
}


/*Read新接口*/
HcclResult CcuKernel::ReadNb(const ChannelHandle channel, const CcuRep::CcuBuf &loc, const CcuRep::RemoteAddr &rem,
                      const CcuRep::Variable &len, CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    Append(std::make_shared<CcuRep::CcuRepBufRead>(channel, rem, loc, len, event, mask));
    return HCCL_SUCCESS;
}

/*Write新接口*/
HcclResult CcuKernel::WriteNb(const ChannelHandle channel, const CcuRep::RemoteAddr &rem, const CcuRep::CcuBuf &loc,
                       const CcuRep::Variable &len, CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    Append(std::make_shared<CcuRep::CcuRepBufWrite>(channel, loc, rem, len, event, mask));
    return HCCL_SUCCESS;
}

static bool isLowPrecisionIn(Hccl::DataType dataType)
{
    return dataType == Hccl::DataType::INT8 || dataType == Hccl::DataType::HIF8 || dataType == Hccl::DataType::FP8E4M3
           || dataType == Hccl::DataType::FP8E5M2;
}

static bool isLowPrecisionOut(Hccl::DataType dataType)
{
    return dataType == Hccl::DataType::FP16 || dataType == Hccl::DataType::BFP16 || dataType == Hccl::DataType::FP32;
}

constexpr uint32_t MAX_DATA_TYPE = 17;

const Hccl::DataType orionDataTypes[] = {
    Hccl::DataType::INT8,
    Hccl::DataType::INT16,
    Hccl::DataType::INT32,
    Hccl::DataType::FP16,
    Hccl::DataType::FP32,
    Hccl::DataType::INT64,
    Hccl::DataType::UINT64,
    Hccl::DataType::UINT8,
    Hccl::DataType::UINT16,
    Hccl::DataType::UINT32,
    Hccl::DataType::FP64,
    Hccl::DataType::BFP16,
    Hccl::DataType::INT128,
#if !defined (OPEN_BUILD_PROJECT) || defined (ORION_MODE)
    Hccl::DataType::HIF8,
    Hccl::DataType::FP8E4M3,
    Hccl::DataType::FP8E5M2,
    Hccl::DataType::FP8E8M0
#endif
};

static Hccl::DataType HcommDataTypeToHcclDataType(const HcclDataType dataType)
{
    const auto dataTypeNum = static_cast<uint32_t>(dataType);
    if (dataTypeNum > MAX_DATA_TYPE) {
        return Hccl::DataType::INVALID;
    }

    return orionDataTypes[dataTypeNum];
}

constexpr uint32_t MAX_REDUCE_TYPE = 4;
const Hccl::ReduceOp orionReduceOps[] = {
    Hccl::ReduceOp::SUM,
    Hccl::ReduceOp::PROD,
    Hccl::ReduceOp::MAX,
    Hccl::ReduceOp::MIN,
};

static Hccl::ReduceOp HcommReduceOpToHcclReduceOp(const HcclReduceOp reduceOp)
{
    const auto reduceOpNum = static_cast<uint32_t>(reduceOp);
    if (reduceOpNum > MAX_REDUCE_TYPE) {
        return Hccl::ReduceOp::INVALID;
    }

    return orionReduceOps[reduceOpNum];
}

HcclResult CcuKernel::LocalReduceNb(const CcuRep::CcuBuf *bufs, uint32_t count, HcclDataType dataType,
                     HcclDataType outputDataType, HcclReduceOp opType,
                     const CcuRep::Variable &len, CcuRep::CompletedEvent event, uint32_t mask)
{
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);
    auto outputDataType_ = HcommDataTypeToHcclDataType(outputDataType);

    if ((opType_ == Hccl::ReduceOp::SUM && isLowPrecisionIn(dataType_) && !isLowPrecisionOut(outputDataType_))
        || (opType_ == Hccl::ReduceOp::SUM && !isLowPrecisionIn(dataType_) && dataType_ != outputDataType_)
        || (opType_ != Hccl::ReduceOp::SUM && dataType_ != outputDataType_)) {
        return HCCL_E_NOT_SUPPORT;
    }

    std::vector<CcuRep::CcuBuf> ccuBufs(count);
    for (uint32_t i = 0; i < count; i++) {
        ccuBufs[i] = bufs[i];
    }

    Append(std::make_shared<CcuRep::CcuRepBufReduce>(ccuBufs, count, CcuRep::GetCcuDataType(dataType_, opType_),
                                                     CcuRep::GetCcuDataType(outputDataType_, opType_),
                                                     CcuRep::GetCcuReduceType(opType_), event, len, mask));
    return HCCL_SUCCESS;
}

/*Read新接口*/
HcclResult CcuKernel::ReadNb(const ChannelHandle channel, const CcuRep::LocalAddr &loc, const CcuRep::RemoteAddr &rem,
                      const CcuRep::Variable &len, CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    Append(std::make_shared<CcuRep::CcuRepRead>(channel, loc, rem, len, event, mask));
    return HCCL_SUCCESS;
}

/*ReadReduce新接口*/
HcclResult CcuKernel::ReadReduceNb(const ChannelHandle channel, const CcuRep::LocalAddr &loc, const CcuRep::RemoteAddr &rem,
                            const CcuRep::Variable &len, HcclDataType dataType, HcclReduceOp opType,
                            CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    Append(std::make_shared<CcuRep::CcuRepRead>(channel, loc, rem, len, CcuRep::GetUBDataType(dataType_),
                                                CcuRep::GetUBReduceType(opType_), event, mask));
    return HCCL_SUCCESS;
}


HcclResult CcuKernel::WriteNb(const ChannelHandle channel, const CcuRep::RemoteAddr &rem, const CcuRep::LocalAddr &loc,
                       const CcuRep::Variable &len, CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    Append(std::make_shared<CcuRep::CcuRepWrite>(channel, rem, loc, len, event, mask));
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::WriteReduceNb(const ChannelHandle channel, const CcuRep::RemoteAddr &rem, const CcuRep::LocalAddr &loc,
                             const CcuRep::Variable &len, HcclDataType dataType, HcclReduceOp opType,
                             CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    Append(std::make_shared<CcuRep::CcuRepWrite>(channel, rem, loc, len, CcuRep::GetUBDataType(dataType_),
                                                 CcuRep::GetUBReduceType(opType_), event, mask));
    return HCCL_SUCCESS;
}

CcuResult CcuKernel::GetBufferByHandle(CcuBufferHandle bufferHandle, CcuRep::CcuBuf **buffer)
{
    return GetResourceByHandle(ccuBufferMap_, bufferHandle, buffer, "buffer");
}

HcclResult CcuKernel::LocalCopyNb(const CcuRep::LocalAddr &dst, const CcuRep::LocalAddr &src, const CcuRep::Variable &len,
                           CcuRep::CompletedEvent event, uint32_t mask)
{
    Append(std::make_shared<CcuRep::CcuRepLocCpy>(dst, src, len, event, mask));
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::LocalCopyNb(const CcuRep::CcuBuf &dst, const CcuRep::LocalAddr &src, const CcuRep::Variable &len,
                           CcuRep::CompletedEvent event, uint32_t mask)
{
    Append(std::make_shared<CcuRep::CcuRepBufLocRead>(src, dst, len, event, mask));
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::LocalCopyNb(const CcuRep::LocalAddr &dst, const CcuRep::CcuBuf &src, const CcuRep::Variable &len,
                           CcuRep::CompletedEvent event, uint32_t mask)
{
    Append(std::make_shared<CcuRep::CcuRepBufLocWrite>(src, dst, len, event, mask));
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::LocalReduceNb(const CcuRep::LocalAddr &dst, const CcuRep::LocalAddr &src, const CcuRep::Variable &len,
                             HcclDataType dataType, HcclReduceOp opType, CcuRep::CompletedEvent event, uint32_t mask)
{
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    Append(std::make_shared<CcuRep::CcuRepLocCpy>(dst, src, len, CcuRep::GetUBDataType(dataType_), CcuRep::GetUBReduceType(opType_),
                                                  event, mask));
    return HCCL_SUCCESS;
}

CcuRep::FuncCall CcuKernel::Func(const std::string &label)
{
    return CcuRep::FuncCall(this, label);
}

CcuRep::FuncCall CcuKernel::Func(const CcuRep::Variable &funcAddr)
{
    return CcuRep::FuncCall(this, funcAddr);
}

CcuRep::LoopCall CcuKernel::Loop(const std::string &label)
{
    return CcuRep::LoopCall(this, label);
}

CcuResult CcuKernel::LoopCreate(CcuLoop *loop)
{
    if (loop == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopCreate] null pointer");
        return CcuResult::CCU_E_PTR;
    }
    if (loopBodyDepth_ > 0) {
        HCCL_ERROR("[CcuKernel::LoopCreate] cannot create loop inside a loop body");
        return CcuResult::CCU_E_INTERNAL;
    }
    if (inFuncBody_) {
        funcBodyError_ = true;
        HCCL_ERROR("[CcuKernel::LoopCreate] cannot create loop inside a func body");
        return CcuResult::CCU_E_INTERNAL;
    }

    CcuLoop handle = ++loopHandleCounter_;
    std::string label = "loop_" + std::to_string(handle);

    LoopDescriptor desc;
    desc.label = label;
    desc.repLoopBlock = std::make_shared<CcuRep::CcuRepLoopBlock>(label);

    loopMap_[handle] = std::move(desc);
    *loop = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopBodyEnter(CcuLoop loop)
{
    auto it = loopMap_.find(loop);
    if (it == loopMap_.end()) {
        HCCL_ERROR("[CcuKernel::LoopBodyEnter] invalid loop handle %lu", loop);
        return CcuResult::CCU_E_PARA;
    }
    auto &desc = it->second;
    if (desc.bodyDefined) {
        HCCL_ERROR("[CcuKernel::LoopBodyEnter] loop %lu body already defined", loop);
        return CcuResult::CCU_E_INTERNAL;
    }

    Append(desc.repLoopBlock);
    desc.prevActiveBlock = CurrentBlock();
    SetCurrentBlock(desc.repLoopBlock);
    ++loopBodyDepth_;

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopBodyExit(CcuLoop loop)
{
    auto it = loopMap_.find(loop);
    if (it == loopMap_.end()) {
        HCCL_ERROR("[CcuKernel::LoopBodyExit] invalid loop handle %lu", loop);
        return CcuResult::CCU_E_PARA;
    }
    auto &desc = it->second;

    SetCurrentBlock(desc.prevActiveBlock);
    desc.bodyDefined = true;
    --loopBodyDepth_;

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncBlockLookup(const void *funcPtr, uint64_t *outHandle)
{
    if (funcPtr == nullptr || outHandle == nullptr) {
        HCCL_ERROR("[CcuKernel::FuncBlockLookup] null pointer");
        return CcuResult::CCU_E_PTR;
    }
    if (loopBodyDepth_ > 0 || inFuncBody_) {
        if (inFuncBody_) {
            funcBodyError_ = true;
        }
        HCCL_ERROR("[CcuKernel::FuncBlockLookup] ccu::CallFunc only allowed at top level");
        return CcuResult::CCU_E_INTERNAL;
    }

    auto it = funcInstanceMap_.find(funcPtr);
    *outHandle = (it == funcInstanceMap_.end()) ? 0 : it->second;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncBlockBegin(const void *funcPtr, uint64_t *outHandle)
{
    if (funcPtr == nullptr || outHandle == nullptr) {
        HCCL_ERROR("[CcuKernel::FuncBlockBegin] null pointer");
        return CcuResult::CCU_E_PTR;
    }
    if (loopBodyDepth_ > 0 || inFuncBody_) {
        if (inFuncBody_) {
            funcBodyError_ = true;
        }
        HCCL_ERROR("[CcuKernel::FuncBlockBegin] ccu::CallFunc only allowed at top level");
        return CcuResult::CCU_E_INTERNAL;
    }

    auto exist = funcInstanceMap_.find(funcPtr);
    if (exist != funcInstanceMap_.end()) {
        *outHandle = exist->second;
        return CcuResult::CCU_SUCCESS;
    }

    const uint64_t handle = ++funcHandleCounter_;
    std::string label = "func_" + std::to_string(handle);

    FuncDescriptor desc;
    desc.funcPtr = funcPtr;
    desc.label = label;
    desc.repFuncBlock = std::make_shared<CcuRep::CcuRepFuncBlock>(label);
    desc.prevActiveBlock = CurrentBlock();

    funcMap_[handle] = desc;
    SetCurrentBlock(desc.repFuncBlock);
    inFuncBody_ = true;
    funcBodyError_ = false;
    *outHandle = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncBlockEnd(uint64_t handle)
{
    auto it = funcMap_.find(handle);
    if (it == funcMap_.end()) {
        HCCL_ERROR("[CcuKernel::FuncBlockEnd] invalid func handle %lu", handle);
        return CcuResult::CCU_E_PARA;
    }
    auto &desc = it->second;

    SetCurrentBlock(desc.prevActiveBlock);
    inFuncBody_ = false;

    if (funcBodyError_) {
        HCCL_ERROR("[CcuKernel::FuncBlockEnd] ccu::CallFunc only allowed at top level");
        funcBodyError_ = false;
        funcMap_.erase(it);
        return CcuResult::CCU_E_INTERNAL;
    }

    Append(desc.repFuncBlock);
    desc.bodyDefined = true;
    funcInstanceMap_[desc.funcPtr] = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncDefineInArg(uint64_t handle, CcuVariableHandle formal)
{
    auto it = funcMap_.find(handle);
    if (it == funcMap_.end()) {
        HCCL_ERROR("[CcuKernel::FuncDefineInArg] invalid func handle %lu", handle);
        return CcuResult::CCU_E_PARA;
    }

    CcuRep::Variable *formalVar = nullptr;
    CCU_CHK_RET(GetVariableByHandle(formal, &formalVar));
    it->second.repFuncBlock->DefineInArg(*formalVar);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncCall(uint64_t handle, const CcuVariableHandle *inArgs, uint32_t numIn)
{
    if (loopBodyDepth_ > 0 || inFuncBody_) {
        if (inFuncBody_) {
            funcBodyError_ = true;
        }
        HCCL_ERROR("[CcuKernel::FuncCall] ccu::CallFunc only allowed at top level");
        return CcuResult::CCU_E_INTERNAL;
    }
    if (numIn > 0 && inArgs == nullptr) {
        HCCL_ERROR("[CcuKernel::FuncCall] null input args");
        return CcuResult::CCU_E_PTR;
    }

    auto it = funcMap_.find(handle);
    if (it == funcMap_.end() || !it->second.bodyDefined) {
        HCCL_ERROR("[CcuKernel::FuncCall] invalid func handle %lu", handle);
        return CcuResult::CCU_E_PARA;
    }

    auto repFuncCall = std::make_shared<CcuRep::CcuRepFuncCall>(it->second.label);
    for (uint32_t i = 0; i < numIn; i++) {
        CcuRep::Variable *actual = nullptr;
        CCU_CHK_RET(GetVariableByHandle(inArgs[i], &actual));
        repFuncCall->SetInArg(*actual);
    }
    Append(repFuncCall);
    return CcuResult::CCU_SUCCESS;
}

// 按 maxLoopNum 把 res_.blockExecutor[0] 扩容到至少 maxLoopNum 个 LoopEngine。
// 与 CreateBlockResAssist 对齐：所有 LoopEngine 资源先落在 die0 池，待
// SelectDie 完成后再由 MoveResourcesToDie 迁移到目标 die。
// 不同 LoopGroup 通过 local loopIdx 复用同一池低位 executorId，所以这里只
// "补足"而不是"累加"。
CcuResult CcuKernel::EnsureLoopEnginePool(uint32_t maxLoopNum)
{
    if (maxLoopNum == 0) {
        HCCL_ERROR("[CcuKernel::EnsureLoopEnginePool] maxLoopNum must be > 0");
        return CcuResult::CCU_E_PARA;
    }
    constexpr uint32_t poolDieId = 0;
    auto &loopEnginePool = res_.blockExecutor[poolDieId];
    if (maxLoopNum <= loopEnginePool.size()) {
        return CcuResult::CCU_SUCCESS;
    }
    const uint32_t deficit = maxLoopNum - static_cast<uint32_t>(loopEnginePool.size());
    std::vector<CcuRep::Executor> tmp(deficit, CcuRep::Executor(this));
    (void)CreateBlockExecutor(deficit, tmp.data());
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupCreate(CcuLoopGroup *group, uint32_t maxLoopNum,
    const CcuLoopGroupConfig *config)
{
    if (group == nullptr || config == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreate] null pointer");
        return CcuResult::CCU_E_PTR;
    }
    if (loopBodyDepth_ > 0) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreate] cannot create loop group inside a loop body");
        return CcuResult::CCU_E_INTERNAL;
    }
    if (inFuncBody_) {
        funcBodyError_ = true;
        HCCL_ERROR("[CcuKernel::LoopGroupCreate] cannot create loop group inside a func body");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 按需扩 LoopEngine 池；池足够则复用低位 executorId，跨组共享。
    CCU_CHK_RET(EnsureLoopEnginePool(maxLoopNum));

    CcuLoopGroup handle = ++loopGroupHandleCounter_;

    LoopGroupDescriptor desc;
    desc.config = *config;
    desc.parallelVar = CreateVariable();
    desc.offsetVar = CreateVariable();
    desc.isVarBased = false;

    auto bundle = std::make_shared<CcuRep::CcuRepLoopGroupBundle>(
        *config, desc.parallelVar, desc.offsetVar);
    desc.bundleRep = bundle;
    Append(bundle);

    loopGroupMap_[handle] = std::move(desc);
    *group = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupCreateFromVar(CcuLoopGroup *group, uint32_t maxLoopNum,
    CcuVariableHandle parallelVarHandle, CcuVariableHandle offsetVarHandle)
{
    if (group == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVar] null pointer for group");
        return CcuResult::CCU_E_PTR;
    }
    if (loopBodyDepth_ > 0) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVar] cannot create loop group inside a loop body");
        return CcuResult::CCU_E_INTERNAL;
    }
    if (inFuncBody_) {
        funcBodyError_ = true;
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVar] cannot create loop group inside a func body");
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(EnsureLoopEnginePool(maxLoopNum));

    CcuRep::Variable *parallelVarPtr = nullptr;
    CcuRep::Variable *offsetVarPtr = nullptr;
    CCU_CHK_RET(GetVariableByHandle(parallelVarHandle, &parallelVarPtr));
    CCU_CHK_RET(GetVariableByHandle(offsetVarHandle, &offsetVarPtr));

    CcuLoopGroup handle = ++loopGroupHandleCounter_;

    LoopGroupDescriptor desc;
    desc.parallelVar = CcuRep::Variable(*parallelVarPtr);
    desc.offsetVar = CcuRep::Variable(*offsetVarPtr);
    desc.isVarBased = true;

    auto bundle = std::make_shared<CcuRep::CcuRepLoopGroupBundle>(
        desc.parallelVar, desc.offsetVar);
    desc.bundleRep = bundle;
    Append(bundle);

    loopGroupMap_[handle] = std::move(desc);
    *group = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupAddLoop(CcuLoopGroup group,
    CcuLoop loop, const CcuLoopConfig *config)
{
    if (config == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoop] null pointer for config");
        return CcuResult::CCU_E_PTR;
    }

    auto grpIt = loopGroupMap_.find(group);
    if (grpIt == loopGroupMap_.end()) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoop] invalid group handle %lu", group);
        return CcuResult::CCU_E_PARA;
    }
    auto &grpDesc = grpIt->second;

    auto loopIt = loopMap_.find(loop);
    if (loopIt == loopMap_.end()) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoop] invalid loop handle %lu", loop);
        return CcuResult::CCU_E_PARA;
    }
    auto &loopDesc = loopIt->second;

    if (!loopDesc.bodyDefined) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoop] loop %lu body not defined", loop);
        return CcuResult::CCU_E_INTERNAL; // CCU_E_LOOP_BODY_UNDEFINED
    }

    // 资源池统一在 die0 上申请，SelectDie 后由 MoveResourcesToDie 迁移
    constexpr uint32_t poolDieId = 0;
    auto &loopEnginePool = res_.blockExecutor[poolDieId];
    const uint32_t loopIdx = grpDesc.loopCount;
    if (loopIdx >= loopEnginePool.size()) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoop] loopEngine pool exhausted "
                   "(pool size %zu, loopIdx %u). Pass a larger maxLoopNum to "
                   "CcuLoopGroupCreate so the pool can be extended at create time.",
                   loopEnginePool.size(), loopIdx);
        return CcuResult::CCU_E_PARA;
    }

    grpDesc.loopCount++;
    grpDesc.totalLoopNum = grpDesc.loopCount;

    CcuRep::CcuRepLoopGroupBundle::LoopEntry entry;
    entry.config = *config;
    entry.executor = loopEnginePool[loopIdx];
    entry.repLoopBlock = loopDesc.repLoopBlock;
    entry.loopParamVar = CreateVariable();
    entry.isVarBased = false;

    auto bundle = std::static_pointer_cast<CcuRep::CcuRepLoopGroupBundle>(grpDesc.bundleRep);
    bundle->AddLoop(entry);

    if (!grpDesc.isVarBased) {
        bundle->SetRepeatLoopIdx(grpDesc.config.cloneLoopOffset);
        bundle->SetTotalLoopNum(grpDesc.totalLoopNum);
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupAddLoopFromVar(CcuLoopGroup group,
    CcuLoop loop, CcuVariableHandle loopParamVarHandle)
{
    auto grpIt = loopGroupMap_.find(group);
    if (grpIt == loopGroupMap_.end()) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoopFromVar] invalid group handle %lu", group);
        return CcuResult::CCU_E_PARA;
    }
    auto &grpDesc = grpIt->second;

    auto loopIt = loopMap_.find(loop);
    if (loopIt == loopMap_.end()) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoopFromVar] invalid loop handle %lu", loop);
        return CcuResult::CCU_E_PARA;
    }
    auto &loopDesc = loopIt->second;

    if (!loopDesc.bodyDefined) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoopFromVar] loop %lu body not defined", loop);
        return CcuResult::CCU_E_INTERNAL; // CCU_E_LOOP_BODY_UNDEFINED
    }

    // 资源池统一在 die0 上申请，SelectDie 后由 MoveResourcesToDie 迁移
    constexpr uint32_t poolDieId = 0;
    auto &loopEnginePool = res_.blockExecutor[poolDieId];
    const uint32_t loopIdx = grpDesc.loopCount;
    if (loopIdx >= loopEnginePool.size()) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoopFromVar] loopEngine pool exhausted "
                   "(pool size %zu, loopIdx %u). Pass a larger maxLoopNum to "
                   "CcuLoopGroupCreateFromVar so the pool can be extended at create time.",
                   loopEnginePool.size(), loopIdx);
        return CcuResult::CCU_E_PARA;
    }

    CcuRep::Variable *loopParamVarPtr = nullptr;
    CCU_CHK_RET(GetVariableByHandle(loopParamVarHandle, &loopParamVarPtr));

    grpDesc.loopCount++;

    CcuRep::CcuRepLoopGroupBundle::LoopEntry entry;
    entry.executor = loopEnginePool[loopIdx];
    entry.repLoopBlock = loopDesc.repLoopBlock;
    entry.loopParamVar = CcuRep::Variable(*loopParamVarPtr);
    entry.isVarBased = true;

    auto bundle = std::static_pointer_cast<CcuRep::CcuRepLoopGroupBundle>(grpDesc.bundleRep);
    bundle->AddLoop(entry);

    return CcuResult::CCU_SUCCESS;
}

void CcuKernel::SetInstrId(uint32_t instrId)
{
    instrInfo_.startInstrId = instrId;
}

uint32_t CcuKernel::GetInstrId() const
{
    return instrInfo_.startInstrId;
}

uint32_t CcuKernel::GetInstrCount()
{
    uint32_t instrCount = 0;
    for (const auto &rep : GetRepSequence()) {
        instrCount += rep->InstrCount();
    }
    instrInfo_.instrCount = instrCount;
    HCCL_INFO("Kernel inst %u", instrCount);
    return instrCount;
}

void CcuKernel::SetCcuInstrInfo(const CcuRep::CcuInstrInfo &instrInfo)
{
    this->instrInfo_ = instrInfo;
}

CcuRep::Variable CcuKernel::CreateVariable()
{
    return CreateResAssist(res_.continuousVariable);
}

CcuRep::Address CcuKernel::CreateAddress()
{
    return CreateResAssist(res_.address);
}

CcuRep::LocalNotify CcuKernel::CreateLocalNotify()
{
    return CreateResAssist(res_.localNotify);
}

CcuRep::CompletedEvent CcuKernel::CreateCompletedEvent()
{
    return CreateResAssist(res_.completedEvent);
}



CcuRep::CcuBuf CcuKernel::CreateCcuBuf()
{
    return CreateResAssist(res_.ccubufs);
}

CcuRep::Executor CcuKernel::CreateExecutor()
{
    return CreateResAssist(res_.executor);
}

CcuRep::LocalAddr CcuKernel::CreateLocalAddr()
{
    return CcuRep::LocalAddr(CreateAddress(), CreateVariable());
}

CcuRep::RemoteAddr CcuKernel::CreateRemoteAddr()
{
    return CcuRep::RemoteAddr(CreateAddress(), CreateVariable());
}

CcuRep::RemoteAddr CcuKernel::GetRemoteAddr(const ChannelHandle channel, uint32_t index)
{
    (void)index;
    channels_.insert(channel);
    auto mem = CcuRep::RemoteAddr(CreateAddress(), CreateVariable());
    Append(std::make_shared<CcuRep::CcuRepRemMem>(channel, mem));
    return mem;
}

CcuRep::LocalAddr CcuKernel::CreateLocalAddr(const CcuRep::Variable &token)
{
    return CcuRep::LocalAddr(CreateAddress(), token);
}

HcclResult CcuKernel::CreateBlockCcuBuf(const uint32_t count, CcuRep::CcuBuf *ccuBufs)
{
    CHK_PTR_NULL(ccuBufs);
    auto resources = CreateBlockResAssist(count, res_.blockCcubufs);

    for (uint32_t i = 0; i < count; i++) {
        ccuBufs[i] = resources[i]; // 拷贝虚拟资源，通过shared_ptr链接到物理资源
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernel::CreateBlockExecutor(const uint32_t count, CcuRep::Executor *ccuExes)
{
    CHK_PTR_NULL(ccuExes);
    auto resources = CreateBlockResAssist(count, res_.blockExecutor);

    for (uint32_t i = 0; i < count; i++) {
        ccuExes[i] = resources[i]; // 拷贝虚拟资源，通过shared_ptr链接到物理资源
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernel::CreateBlockCompletedEvent(const uint32_t count, CcuRep::CompletedEvent *ccuEvents)
{
    CHK_PTR_NULL(ccuEvents);
    auto resources = CreateBlockResAssist(count, res_.blockCompletedEvent);

    for (uint32_t i = 0; i < count; i++) {
        ccuEvents[i] = resources[i]; // 拷贝虚拟资源，通过shared_ptr链接到物理资源
    }

    return HcclResult::HCCL_SUCCESS;
}


void CcuKernel::SetResRepository(const CcuResRepository &resRepo)
{
    resRepo_ = resRepo;
}

CcuResRepository  &CcuKernel::GetResRepository()
{
    return resRepo_;
}

CcuSharedResource &CcuKernel::GetExportedRes()
{
    return exportedRes_;
}

CcuSharedResource &CcuKernel::GetImportedRes()
{
    return importedRes_;
}

static HcclResult GetArgIndex(const std::unordered_map<uint16_t, uint16_t> &varId2VarIdMap,
                                 const std::unordered_map<uint16_t, uint32_t> &varId2ArgIndexMap,
                                 const uint64_t *taskArgs, uint32_t argSize,
                                 uint16_t varId, uint64_t& argIndex)
{
    HCCL_INFO("[GetArgIndex] Enter varId(%u)", varId);
    auto item = varId2ArgIndexMap.find(varId);
    if (item == varId2ArgIndexMap.end()) {
        uint16_t oriVarId = varId;
        auto iter = varId2VarIdMap.find(varId);
        while (iter != varId2VarIdMap.end()) { // 循环查找中间assign Rep，找到起始varId
            oriVarId = iter->second;
            iter = varId2VarIdMap.find(oriVarId);
        }
        if (oriVarId != varId) { // 起始varId预期通过LoadArg赋值
            item = varId2ArgIndexMap.find(oriVarId);
            if (item == varId2ArgIndexMap.end()) {
                HCCL_ERROR("[%s]fail, Invalid goSize variable id(%u), oriVarId = %u", __func__, varId, oriVarId);
                return HCCL_E_PARA;
            }
        } else {
            HCCL_ERROR("[%s]fail, Invalid goSize variable id(%u)", __func__, varId);
            return HCCL_E_PARA;
        }
    }
    HCCL_INFO("[GetArgIndex] find end");
    if (item->second >= argSize) {
        HCCL_ERROR("Invalid goSize variable index(%u).", item->second);
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "GetArgIndex success: varId(%u) varId2VarIdMapSize(%u) varId2ArgIndexMapSize(%u) taskArgsSize(%u)",
        varId, varId2VarIdMap.size(), varId2ArgIndexMap.size(), argSize);
    argIndex = taskArgs[item->second];
    return HCCL_SUCCESS;
}

void DumpCcuProfilingInfo(const std::vector<CcuProfilingInfo> &ccuProfilingInfo)
{
    auto dumpLinkInfo = [] (const CcuProfilingInfo &info) -> void {
        for (int i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
            if (info.channelId[i] == INVALID_VALUE_CHANNELID) {
                continue;
            }
            HCCL_INFO("channelId(%u), remoteRankId(%u).", info.channelId[i], info.remoteRankId[i]);
        }
    };

    for (const auto &profInfo : ccuProfilingInfo) {
        if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING)) {
            HCCL_INFO("Dump CCU Profiling Info:SQE Profiling Info: ctxSignautre(%s), "
                       "dieId(%d), missionId(%d), instrId(%d).",
                       profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                       static_cast<int>(profInfo.instrId));
        } else if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_WAITCKE_PROFILING)) {
            HCCL_INFO("Microcode WaitCKE Profiling Info: name(%s), "
                       "dieId(%d), missionId(%d), instrId(%d), ckeId(%u), mask(%u).",
                       profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                       static_cast<int>(profInfo.instrId), profInfo.ckeId, profInfo.mask);
            dumpLinkInfo(profInfo);
        } else if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_LOOPGROUP_PROFILING)) {
            HCCL_INFO("Microcode LoopGroup Profiling Info: name(%s), "
                       "dieId(%d), missionId(%d), instrId(%d), reduceOpType(%d), inputDataType(%d), "
                       "outputDataType(%d), dataSize(%llu).",
                       profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                       static_cast<int>(profInfo.instrId), static_cast<int>(profInfo.reduceOpType),
                       static_cast<int>(profInfo.inputDataType), static_cast<int>(profInfo.outputDataType),
                       profInfo.dataSize);
            dumpLinkInfo(profInfo);
        }
    }
}

constexpr uint64_t SetBits(uint16_t end)
{
    return ((uint64_t(1) << (end + 1)) - uint64_t(1));
}

static uint16_t ParseRepeatNumFromParallelParam(uint64_t parallelParam)
{
    constexpr uint16_t repeatBitNum       = 7; // 7： repeat num 占 7 bits
    constexpr uint16_t repeatNumShiftBit  = 55; // 55： repeat num占[61:55]位置
    return ( parallelParam >> repeatNumShiftBit) & SetBits(repeatBitNum);
}

HcclResult CcuKernel::CollectSqeAndWaitCkeProfilingInfo()
{
    auto &ccuProfilingCache = GetProfilingInfo();
    uint32_t count {0};
    HCCL_INFO("[GetCcuProfilingInfo] Process sqe&waitcke profiling info start.");
    for (auto &profInfo : ccuProfilingCache) {
        profInfo.missionId = GetMissionId();
        if (profInfo.type == static_cast<uint8_t>(hcomm::CcuProfilinType::CCU_TASK_PROFILING)) {
            profInfo.instrId   = GetInstrId();
            allCcuProfilingInfos_.push_back(profInfo);
            continue;
        }
        if (count >= GetWaiteCkeProfilingReps().size()) {
            HCCL_ERROR("count[%u] out of range[0, %u], cache size(%u).", count,
                GetWaiteCkeProfilingReps().size(), ccuProfilingCache.size());
            return HCCL_E_INTERNAL;
        }
        auto waitCkeRep = GetWaiteCkeProfilingReps()[count];
        profInfo.instrId = waitCkeRep->StartInstrId();
        if (profInfo.ckeId == INVALID_CKE_ID) { // localWait Rep
            if (waitCkeRep.get() == nullptr) {
                HCCL_ERROR("[GetCcuProfilingInfo] localWaitRep is nullptr.");
                return HCCL_E_PTR;
            }
            profInfo.ckeId = waitCkeRep->GetId();
            HCCL_INFO("[CcuKernel][GetCcuProfilingInfo] waitcke[%u]", profInfo.ckeId);
        }
        allCcuProfilingInfos_.push_back(profInfo);
        count++;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::BuildLoopGroupVarIdMaps(std::unordered_map<uint16_t, uint32_t> &varId2ArgIndexMap,
    std::unordered_map<uint16_t, uint16_t> &varId2VarIdMap)
{
    auto &lgProfInfo = GetLGProfilingInfo();
    HCCL_INFO("[GetCcuProfilingInfo] create varId2ArgIndexMap start. size=%lu",
        lgProfInfo.loadRep2ArgIdxMap.size());
    for (auto &iter : lgProfInfo.loadRep2ArgIdxMap) {
        if (iter.first.get() == nullptr) {
            HCCL_ERROR("[GetCcuProfilingInfo] loadRep is nullptr.");
            return HCCL_E_PTR;
        }
        auto loadRep = dynamic_cast<CcuRep::CcuRepLoadArg*>(iter.first.get());
        varId2ArgIndexMap[loadRep->GetVarId()] = iter.second;
    }

    HCCL_INFO("[GetCcuProfilingInfo] create varId2VarIdMap start. size=%lu",
        lgProfInfo.assignProfilingReps.size());
    for (auto &iter : lgProfInfo.assignProfilingReps) {
        if (iter.get() == nullptr) {
            HCCL_ERROR("[GetCcuProfilingInfo] assignRep is nullptr.");
            return HCCL_E_PTR;
        }
        auto assignRep = dynamic_cast<CcuRep::CcuRepAssign*>(iter.get());
        varId2VarIdMap[assignRep->varB.Id()] = assignRep->varA.Id();
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::CollectLoopGroupProfilingInfo(const uint64_t *taskArgs, uint32_t argSize,
    const std::unordered_map<uint16_t, uint32_t> &varId2ArgIndexMap,
    const std::unordered_map<uint16_t, uint16_t> &varId2VarIdMap)
{
    auto &lgProfInfo = GetLGProfilingInfo();
    HCCL_INFO("[GetCcuProfilingInfo] process loop group profiling start: "
        "lgsize(%lu), goSize(%lu)", lgProfInfo.lgProfilingReps.size(), groupOpSizeInfo_.size());
    for (uint32_t i = 0; i < lgProfInfo.lgProfilingReps.size(); i += 2) { // 2: 一个goSize对应一个CcuProfilingInfo，对应1个loopGroup Rep
        if (argSize == 0 || varId2ArgIndexMap.empty()) {
            continue;
        }
        uint64_t loopParam {0};
        CHK_RET(GetArgIndex(varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize,
            groupOpSizeInfo_[i].loopParamId, loopParam));
        uint64_t parallelParam {0};
        CHK_RET(GetArgIndex(varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize,
            groupOpSizeInfo_[i].parallelParamId, parallelParam));
        HCCL_INFO("Collect loopgroup profiling info: repSize[%u], index[%u],"
            "loopParam[%llu], parallelParam[%llu].",
            lgProfInfo.lgProfilingReps.size(), i, loopParam, parallelParam);

        if (loopParam != 0) {
            lgProfInfo.ccuProfilingInfos[i].dataSize = loopParam * moConfig_.loopCount * moConfig_.memSlice;
            lgProfInfo.ccuProfilingInfos[i].instrId = dynamic_cast<CcuRep::CcuRepLoopGroupBundle*>(lgProfInfo.lgProfilingReps[i].get())->StartInstrId();
            allCcuProfilingInfos_.push_back(lgProfInfo.ccuProfilingInfos[i]);
        }

        if (parallelParam != 0) {
            HCCL_INFO("[GetCcuProfilingInfo] collect lg, residual start i=%lu", i);
            uint64_t residual {0};
            CHK_RET(GetArgIndex(varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize,
                groupOpSizeInfo_[i].residualId, residual));
            uint64_t repeatNum = ParseRepeatNumFromParallelParam(parallelParam);
            lgProfInfo.ccuProfilingInfos[i].dataSize = repeatNum * moConfig_.memSlice + residual;
            lgProfInfo.ccuProfilingInfos[i].instrId = dynamic_cast<CcuRep::CcuRepLoopGroupBundle*>(lgProfInfo.lgProfilingReps[i + 1].get())->StartInstrId();
            allCcuProfilingInfos_.push_back(lgProfInfo.ccuProfilingInfos[i]);
        }
    }
    return HCCL_SUCCESS;
}

/*
 	* variable/maskSignal等资源变量Id，一定要在获取ccu profiling时才获取；
 	* 原因：在创建context Rep时，其资源Id属于虚拟资源；翻译时，才会绑定固定的物理资源。
*/
HcclResult CcuKernel::GetCcuProfilingInfo(const uint64_t *taskArgs, uint32_t argSize,
    std::vector<CcuProfilingInfo> &allCcuProfilingInfo)
{
    HCCL_INFO("[GetCcuProfilingInfo] Enter.");
    allCcuProfilingInfos_.clear();

    CHK_RET(CollectSqeAndWaitCkeProfilingInfo());

    std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap;
    std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
    CHK_RET(BuildLoopGroupVarIdMaps(varId2ArgIndexMap, varId2VarIdMap));

    CHK_RET(CollectLoopGroupProfilingInfo(taskArgs, argSize, varId2ArgIndexMap, varId2VarIdMap));

    DumpCcuProfilingInfo(allCcuProfilingInfos_);
    allCcuProfilingInfo = allCcuProfilingInfos_;
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::AddProfilingInfo(const ChannelHandle *channels, uint32_t channelNum, HcclDataType dataType,
                                HcclDataType outputDataType, HcclReduceOp opType, const std::string& opName)
{
    CHK_PTR_NULL(channels);
    ccuProfilingInfoCache.type           = (uint8_t)CcuProfilinType::CCU_LOOPGROUP_PROFILING;
    ccuProfilingInfoCache.name           = opName;
    ccuProfilingInfoCache.reduceOpType   = opType;
    ccuProfilingInfoCache.inputDataType  = dataType;
    ccuProfilingInfoCache.outputDataType = outputDataType;
    ccuProfilingInfoCache.missionId      = GetMissionId();
    
    CHK_SAFETY_FUNC_RET(memset_s(ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId),
                                    INVALID_VALUE_CHANNELID, sizeof(ccuProfilingInfoCache.channelId)));
    for (uint32_t i = 0; i < channelNum; i++) {
        void *channelPtr{nullptr};
        CHK_RET(static_cast<HcclResult>(HcommChannelGet(channels[i], &channelPtr)));
        auto *channelImpl = dynamic_cast<CcuUrmaChannel *>(static_cast<Channel *>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        ccuProfilingInfoCache.channelId[i] = channelImpl->GetChannelId();
        ccuProfilingInfoCache.channelHandle[i] = channels[i];
        HCCL_INFO("[%s]type[%d], name[%s], opType[%d], dataType[%d], outputDataType[%d], missionId[%u], "
                "channelHandle[0x%llx], channelId[%u]", __func__, ccuProfilingInfoCache.type, 
                ccuProfilingInfoCache.name.c_str(), opType, dataType, outputDataType, ccuProfilingInfoCache.missionId,
                ccuProfilingInfoCache.channelHandle[i], ccuProfilingInfoCache.channelId[i]);
    }
    lgProfilingInfo.ccuProfilingInfos.push_back(ccuProfilingInfoCache);
    lgProfilingInfo.lgProfilingReps.push_back(allLgProfilingReps.back());
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::AddCcuProfiling(GroupInfo groupInfo, const std::vector<ChannelHandle> channelHandle, HcclDataType dataType,
                                 HcclDataType outputDataType, HcclReduceOp opType, const std::string& opName)
{
    CHK_RET(AddCcuProfiling(channelHandle.data(), channelHandle.size(), dataType, outputDataType, opType, opName));
    groupOpSizeInfo_.push_back(groupInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::AddCcuProfiling(const ChannelHandle *channels, uint32_t channelNum, HcclDataType dataType,
                                HcclDataType outputDataType, HcclReduceOp opType, const std::string& opName)
{
    CHK_PTR_NULL(channels);
    CHK_RET(AddProfilingInfo(channels, channelNum, dataType, outputDataType, opType, opName));
    return HCCL_SUCCESS;
}

}; // namespace hcomm