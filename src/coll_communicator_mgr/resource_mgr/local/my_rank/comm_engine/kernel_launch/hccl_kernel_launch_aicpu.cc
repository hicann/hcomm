/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <sstream>
#include <string>
#include <vector>

#include "hccl/hccl_launch.h"
#include "acl/acl_rt.h"
#include "hccl_group.h"
#include "hccl_res_expt.h"
#include "hccl_aicpu_interface.h"

#include "hccl_independent_common.h"
#include "group_schedule_mgr.h"

using namespace hccl;

constexpr uint32_t NUM_ZERO = 0;
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
static uint32_t g_KernelLaunchTimeout = UINT16_MAX;

static HcclResult
LaunchNotifyWaitToThread(HcclComm comm, aclrtStream unfoldStream, ThreadHandle srcThread, uint32_t dstNotifyIdx)
{
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    std::string kernelName = "RunAicpuNotifyWait";
    // 1. 获取 function handle
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto binKernelHandle = hcclComm->GetBinHandle();
    aclError ret = aclrtBinaryGetFunction(binKernelHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, "
            "kernelName:%s",
            ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    // 2. 初始化 args handle
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    // 3. 准备参数并 append
    ThreadNotifyWaitParam param;
    param.thread = srcThread;
    param.notifyIdx = dstNotifyIdx;
    aclrtParamHandle paraHandle;
    ret = aclrtKernelArgsAppend(argsHandle, &param, sizeof(ThreadNotifyWaitParam), &paraHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, kernelName:%s", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    // 4. finalize args
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    // 5. 下发 kernel
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = g_KernelLaunchTimeout;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;

    ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &cfg, argsHandle, nullptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed, "
            "kernelName:%s",
            ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

static HcclResult LaunchP2pExec(
    HcclComm comm, aclrtStream unfoldStream, const HcclKernelFuncInfo* funcInfo, const void* funcArgs, uint32_t argSize,
    ThreadHandle sendRecvThread)
{
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    // 1. 获取 function handle
    aclrtBinHandle binKernelHandle = nullptr;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    CHK_RET(collComm->GetHcclBinHandle(binKernelHandle));
    aclError ret = aclrtBinaryGetFunction(binKernelHandle, funcInfo->kernelFuncName, &funcHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s", ret,
            funcInfo->kernelFuncName),
        HCCL_E_RUNTIME);

    // 2. 初始化 args handle
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s", ret, funcInfo->kernelFuncName),
        HCCL_E_RUNTIME);

    // 3. 准备参数并 append
    HcclP2pKernelParam params;
    params.sendRecvThread = sendRecvThread;
    memset_s(params.opParams, P2P_MAX_ARG_SIZE, 0, P2P_MAX_ARG_SIZE);
    memcpy_s(params.opParams, P2P_MAX_ARG_SIZE, funcArgs, argSize);

    aclrtParamHandle paraHandle;
    ret = aclrtKernelArgsAppend(argsHandle, &params, sizeof(HcclP2pKernelParam), &paraHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, kernelName:%s", ret, funcInfo->kernelFuncName),
        HCCL_E_RUNTIME);

    // 4. finalize args
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s", ret,
            funcInfo->kernelFuncName),
        HCCL_E_RUNTIME);

    // 5. 下发 kernel
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = g_KernelLaunchTimeout;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;

    ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &cfg, argsHandle, nullptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed, "
            "kernelName:%s",
            ret, funcInfo->kernelFuncName),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

// 放到kernel launch的地方
static HcclResult LaunchNotifyRecordToThread(
    HcclComm comm, aclrtStream unfoldStream, ThreadHandle srcThread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    std::string kernelName = "RunAicpuNotifyRecord";

    // 1. 获取 function handle
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto binKernelHandle = hcclComm->GetBinHandle();
    aclError ret = aclrtBinaryGetFunction(binKernelHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    // 2. 初始化 args handle
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    // 3. 准备参数并 append
    ThreadNotifyRecordParam param;
    param.thread = srcThread;
    param.dstThread = dstThread;
    param.dstNotifyIdx = dstNotifyIdx;
    aclrtParamHandle paraHandle;
    ret = aclrtKernelArgsAppend(argsHandle, &param, sizeof(ThreadNotifyRecordParam), &paraHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, kernelName:%s", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    // 4. finalize args
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    // 5. 下发 kernel
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = g_KernelLaunchTimeout;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;

    ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &cfg, argsHandle, nullptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed, "
            "kernelName:%s",
            ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

static HcclResult AicpuKernelLaunchDirect(
    HcclComm comm, const HcclKernelFuncInfo* funcInfo, ThreadHandle aicpuThreadHandle, aclrtStream unfoldStream,
    aclrtStream userStream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(unfoldStream);
    CHK_PTR_NULL(userStream);
    CHK_PTR_NULL(funcInfo);

    void* args = funcInfo->args;
    uint32_t argSize = funcInfo->argSize;
    if (argSize > 0 && args == nullptr) {
        HCCL_ERROR("[AicpuKernelLaunchDirect] args is null but argSize[%u] > 0", argSize);
        return HCCL_E_PTR;
    }

    ThreadHandle cpuTsThread{0};
    ThreadHandle exportedAicpuTsThread{0};
    ThreadHandle exportedCpuTsThread{0};
    uint32_t notifyNumOnMainThread;
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, userStream, NUM_THREE, &cpuTsThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, NUM_ONE, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, NUM_ONE, &aicpuThreadHandle, COMM_ENGINE_CPU_TS, &exportedCpuTsThread));
    CHK_RET(HcclGetNotifyNumInThread(comm, exportedCpuTsThread, COMM_ENGINE_AICPU_TS, &notifyNumOnMainThread));

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread - 1))); // h2d record
    uint64_t beginTime = HcommGetProfilingSysCycleTime(); // AicpuKernel report start
    CHK_RET(LaunchNotifyWaitToThread(comm, unfoldStream, aicpuThreadHandle, notifyNumOnMainThread - 1)); // device wait
    CHK_RET(LaunchP2pExec(comm, unfoldStream, funcInfo, args, argSize, aicpuThreadHandle)); // device run task
    CHK_RET(LaunchNotifyRecordToThread(
        comm, unfoldStream, aicpuThreadHandle, exportedAicpuTsThread, NUM_ZERO)); // d2h record
    std::string kernelNameCStr(funcInfo->kernelFuncName);
    HcclResult ret = HcclReportAicpuKernel(comm, beginTime, kernelNameCStr.data()); // AicpuKernel report end
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AicpuKernelLaunchDirect] HcclReportAicpuKernel failed, beginTime %llu, kernelName %s, ret %d ", beginTime,
            kernelNameCStr.c_str(), ret);
        return ret;
    }
    CHK_RET(
        static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(cpuTsThread, NUM_ZERO))); // host wait

    return HCCL_SUCCESS;
}

HcclResult HcclAicpuKernelLaunch(
    HcclComm comm, const HcclOpDesc* opInfo, const HcclKernelFuncInfo* funcInfo, ThreadHandle aicpuThreadHandle,
    aclrtStream userStream, const HcclKernelLaunchCfg* kernelLaunchCfg)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(userStream);
    CHK_PTR_NULL(funcInfo);
    CHK_PTR_NULL(opInfo);
    CHK_PTR_NULL(kernelLaunchCfg);

    uint32_t argSize = funcInfo->argSize;
    void* args = funcInfo->args;

    g_KernelLaunchTimeout = kernelLaunchCfg->timeOut;
    if (argSize > 0 && args == nullptr) {
        HCCL_ERROR("[HcclAicpuKernelLaunch] args is null but argSize[%u] > 0", argSize);
        return HCCL_E_PTR;
    }

    HCCL_INFO(
        "[HcclAicpuKernelLaunch] opDescType[%u], kernelSo[%s], kernelFuncName[%s], argSize[%u], "
        "aicpuThreadHandle[%llu], hcclGroupDepth[%d]",
        opInfo->opDescType, funcInfo->kernelSoName, funcInfo->kernelFuncName, argSize, aicpuThreadHandle,
        hcclGroupDepth);

    if (hcclGroupDepth > 0) {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        if (argSize > P2P_MAX_ARG_SIZE) {
            HCCL_ERROR("[HcclAicpuKernelLaunch] argSize[%u] over P2P_MAX_ARG_SIZE", argSize);
            return HCCL_E_PARA;
        }
        HCCL_INFO("[HcclAicpuKernelLaunch] group mode, add p2p task hcclGroupDepth[%d]", hcclGroupDepth);
        HcclP2pTask task;
        task.desc = opInfo->p2p;
        task.stream = opInfo->p2p.unfoldStream;
        memcpy_s(
            task.funcInfo.kernelSoName, HCCL_KERNEL_SO_NAME_MAX_LEN, funcInfo->kernelSoName,
            HCCL_KERNEL_SO_NAME_MAX_LEN);
        memcpy_s(
            task.funcInfo.kernelFuncName, HCCL_KERNEL_FUNC_NAME_MAX_LEN, funcInfo->kernelFuncName,
            HCCL_KERNEL_FUNC_NAME_MAX_LEN);
        memcpy_s(task.args, P2P_MAX_ARG_SIZE, args, argSize);
        task.argSize = argSize;
        task.usrStream = userStream;
        CHK_RET(collComm->groupScheduleMgr->AppendGroupP2pTask(comm, task, opInfo->p2p));
        return HCCL_SUCCESS;
    }

    return AicpuKernelLaunchDirect(comm, funcInfo, aicpuThreadHandle, opInfo->p2p.unfoldStream, userStream);
}

static HcclResult GetStreams(
    const CollComm* collComm, const std::vector<HcclP2pTask>& sortedSendQue,
    const std::vector<HcclP2pTask>& sortedRecvQue, aclrtStream& unfoldStream, const aclrtStream& usrStream)
{
    if (!sortedSendQue.empty()) {
        unfoldStream = sortedSendQue[0].stream;
        CHK_RET(collComm->groupScheduleMgr->SetUsrStream(sortedSendQue[0].usrStream));
    } else if (!sortedRecvQue.empty()) {
        unfoldStream = sortedRecvQue[0].stream;
        CHK_RET(collComm->groupScheduleMgr->SetUsrStream(sortedRecvQue[0].usrStream));
    } else {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult groupLaunchA5()
{
    std::vector<HcclComm> hcclGroupCommListV2 = GetHcclGroupCommList();
    HCCL_INFO("[groupLaunchA5] to the start hcclGroupCommListV2.size[%zu]", hcclGroupCommListV2.size());

    for (HcclComm comm : hcclGroupCommListV2) {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);

        /*新建send/recv流*/
        ThreadHandle sendRecv[2];
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, NUM_TWO, NUM_ONE, sendRecv));
        ThreadHandle aicpuSendThread = sendRecv[0], aicpuRecvThread = sendRecv[1];
        std::vector<HcclP2pTask> sortedSendQue, sortedRecvQue;
        CHK_RET(collComm->groupScheduleMgr->GetP2pTaskSchedule(sortedSendQue, sortedRecvQue));
        aclrtStream unfoldStream = nullptr, usrStream = nullptr;
        ThreadHandle cpuTsThread = 0, exportedAicpuTsThread = 0, exportedCpuTsSendThread = 0,
                     exportedCpuTsRecvThread = 0;
        CHK_RET(GetStreams(collComm, sortedSendQue, sortedRecvQue, unfoldStream, usrStream));

        CHK_RET(collComm->groupScheduleMgr->GetUsrStream(usrStream));
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, usrStream, NUM_THREE, &cpuTsThread));
        CHK_RET(
            HcclThreadExportToCommEngine(comm, NUM_ONE, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
        CHK_RET(HcclThreadExportToCommEngine(
            comm, NUM_ONE, &aicpuSendThread, COMM_ENGINE_CPU_TS, &exportedCpuTsSendThread));
        CHK_RET(HcclThreadExportToCommEngine(
            comm, NUM_ONE, &aicpuRecvThread, COMM_ENGINE_CPU_TS, &exportedCpuTsRecvThread));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsSendThread, NUM_ZERO)));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsRecvThread, NUM_ZERO)));

        // 下发wait kernel
        CHK_RET(LaunchNotifyWaitToThread(comm, unfoldStream, aicpuSendThread, NUM_ZERO));
        CHK_RET(LaunchNotifyWaitToThread(comm, unfoldStream, aicpuRecvThread, NUM_ZERO));

        // Send/Recv交替执行以避免死锁
        for (size_t sendIdx = 0, recvIdx = 0; sendIdx < sortedSendQue.size() || recvIdx < sortedRecvQue.size();) {
            if (sendIdx < sortedSendQue.size()) {
                CHK_RET(LaunchP2pExec(
                    comm, sortedSendQue[sendIdx].stream, &sortedSendQue[sendIdx].funcInfo, sortedSendQue[sendIdx].args,
                    sortedSendQue[sendIdx].argSize, aicpuSendThread));
                sendIdx++;
            }
            if (recvIdx < sortedRecvQue.size()) {
                CHK_RET(LaunchP2pExec(
                    comm, sortedRecvQue[recvIdx].stream, &sortedRecvQue[recvIdx].funcInfo, sortedRecvQue[recvIdx].args,
                    sortedRecvQue[recvIdx].argSize, aicpuRecvThread));
                recvIdx++;
            }
        }

        // 下发record kernel
        CHK_RET(LaunchNotifyRecordToThread(comm, unfoldStream, aicpuSendThread, exportedAicpuTsThread, NUM_ONE));
        CHK_RET(LaunchNotifyRecordToThread(comm, unfoldStream, aicpuRecvThread, exportedAicpuTsThread, NUM_TWO));

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(cpuTsThread, NUM_ONE)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(cpuTsThread, NUM_TWO)));
    }

    SetHcclP2pTaskNums(0);
    ClearHcclGroupCommList();

    return HCCL_SUCCESS;
}
