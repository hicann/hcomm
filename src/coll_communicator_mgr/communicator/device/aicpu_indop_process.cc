/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_indop_process.h"
#include "coll_comm_aicpu_mgr.h"
#include "hcclCommOp.h"
#include "hcclCommDfxLite.h"

using namespace hccl;

namespace {
struct CollCommAicpuInfo {
    ReadWriteLockBase commAicpuMgrMapMutex;  // 读写锁单例，维护全局的读写信息
    std::unordered_map<std::string, std::unique_ptr<CollCommAicpuMgr>> commMgrMap;
};
CollCommAicpuInfo g_commAicpuInfo;

thread_local CollCommAicpuMgr *g_hcclComm = nullptr; // 记录当前线程通信域; AicpuGetCommbyGroup赋值，AicpuReleaseCommMgrbyGroup置空
}

HcclResult AicpuIndopProcess::AicpuIndOpCommInit(CommAicpuParam *commAicpuParam) {
    CHK_PTR_NULL(commAicpuParam);

    CollCommAicpuMgr *commAicpuMgr = nullptr;
    HcclResult ret = HCCL_SUCCESS;
    std::string group = commAicpuParam->hcomId;
    CHK_RET(AcquireAicpuCommMgr(group, &commAicpuMgr));
    if (commAicpuMgr == nullptr) {
        HCCL_ERROR("[AicpuIndopProcess][AicpuIndOpCommInit]commAicpu is null group[%s]", group.c_str());
        return HCCL_E_PTR;
    }

    ret = commAicpuMgr->InitAicpuIndOp(commAicpuParam);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuIndopProcess][%s]errNo[0x%016llx] Failed to init independent op comm group[%s]" , __func__,
        HCCL_ERROR_CODE(ret), group.c_str()), ret);

    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::AcquireAicpuCommMgr(const std::string &group, CollCommAicpuMgr **aicpuCommMgrPtr)
{
    ReadWriteLock rwlock(g_commAicpuInfo.commAicpuMgrMapMutex);
    rwlock.writeLock();
    // 查找是否已存在该group的通信实例
    auto iter = g_commAicpuInfo.commMgrMap.find(group);
    if (iter != g_commAicpuInfo.commMgrMap.end()) {
        *aicpuCommMgrPtr = iter->second.get();
        // 创建aicpu通信域
        CHK_RET(iter->second->AcquireCollCommAicpu());
        HCCL_INFO("[%s]Reuse existing comm group [%s]", __func__, group.c_str());
        rwlock.writeUnlock();
        return HCCL_SUCCESS;
    }
    
    // 未找到则创建新实例
    std::unique_ptr<CollCommAicpuMgr> aicpuCommMgr;
    EXCEPTION_CATCH(aicpuCommMgr = std::make_unique<CollCommAicpuMgr>(), return HCCL_E_PTR);
    // 创建aicpu通信域
    CHK_RET(aicpuCommMgr->AcquireCollCommAicpu());

    // 将新实例加入映射表
    *aicpuCommMgrPtr = aicpuCommMgr.get();
    g_commAicpuInfo.commMgrMap.insert({group, std::move(aicpuCommMgr)});
    HCCL_RUN_INFO("[%s]Created new comm group [%s]", __func__, group.c_str());
    rwlock.writeUnlock();
    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::AicpuIndOpThreadInit(ThreadMgrAicpuParam *param)
{
    CHK_PTR_NULL(param);

    std::string group = param->hcomId;
    HCCL_INFO("[%s]group[%s]", __func__, group.c_str());
    CollCommAicpuMgr *collCommAicpuMgr = AicpuIndopProcess::AicpuGetCommMgrbyGroup(group);
    CHK_PRT_RET(collCommAicpuMgr == nullptr, HCCL_ERROR("%s collCommAicpuMgr is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);
    HcclResult ret = collCommAicpuMgr->InitThreads(param);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuIndopProcess][AicpuIndOpThreadInit]errNo[0x%016llx] Failed to init threads group[%s]",
        HCCL_ERROR_CODE(ret), group.c_str()), ret);
    AicpuReleaseCommMgrbyGroup(group);
    return HCCL_SUCCESS;
}

CollCommAicpuMgr *AicpuIndopProcess::AicpuGetCommMgrbyGroup(const std::string &group)
{
    HCCL_INFO("[AicpuIndopProcess][%s]start, group[%s]", __func__, group.c_str());
    auto startTime = std::chrono::steady_clock::now();
    constexpr u32 pollIntervalUs = 10; // 轮询间隔10us
    constexpr u32 pollTimeoutMs = 10000; //临时规避host侧在临时流下发kernel获取锁超时的问题
    auto waitPollTimeOutMs = std::chrono::milliseconds(pollTimeoutMs);
    ReadWriteLock rwlock(g_commAicpuInfo.commAicpuMgrMapMutex);

    while (true) {
        rwlock.readLock();
        auto iter = g_commAicpuInfo.commMgrMap.find(group);
        if (iter == g_commAicpuInfo.commMgrMap.end()) {
            HCCL_ERROR("[AicpuIndopProcess] exist group size is [%u]", g_commAicpuInfo.commMgrMap.size());
            auto curIter = g_commAicpuInfo.commMgrMap.begin();
            int i = 0;
            while (curIter != g_commAicpuInfo.commMgrMap.end()) {
                HCCL_ERROR("[AicpuIndopProcess] exist group idx is [%d] key[%s] value", i, curIter->first.c_str());
                curIter++;
            }
            rwlock.readUnlock();
            return nullptr;
        }
        if (iter->second->IsUsed()) {
            if ((std::chrono::steady_clock::now() - startTime) >= waitPollTimeOutMs) {
                HCCL_ERROR("[AicpuIndopProcess][%s]poll timeout, comm group [%s] has been used", __func__, group.c_str());
                rwlock.readUnlock();
                return nullptr;
            }
            rwlock.readUnlock();
            usleep(pollIntervalUs);
            continue;
        }
        g_hcclComm = iter->second.get();
        iter->second->SetUsed(true);
        HCCL_INFO("[AicpuIndopProcess][%s]success, group[%s]", __func__, group.c_str());
        rwlock.readUnlock();
        return iter->second.get();
    }
    return nullptr;
}

void AicpuIndopProcess::AicpuReleaseCommMgrbyGroup(const std::string &group)
{
    ReadWriteLock rwlock(g_commAicpuInfo.commAicpuMgrMapMutex);
    rwlock.readLock();
    auto iter = g_commAicpuInfo.commMgrMap.find(group);
    if (iter == g_commAicpuInfo.commMgrMap.end()) {
        rwlock.readUnlock();
        return;
    }
    g_hcclComm = nullptr;
    iter->second->SetUsed(false);
    rwlock.readUnlock();
}

CollCommAicpuMgr *AicpuIndopProcess::AicpuGetComm(const std::string &group)
{
    if (group.empty()) {
        HCCL_ERROR("[AicpuIndopProcess][%s] comm group is empty", __func__);
        return nullptr;
    }
    if (g_hcclComm == nullptr) {
        HCCL_ERROR("[AicpuIndopProcess][%s] g_hcclComm is nullptr", __func__);
        return nullptr;
    }

    if (g_hcclComm->GetCollCommAicpu()->GetIdentifier() != group) {
        HCCL_ERROR("[AicpuIndopProcess][%s] comm group[%s] is not current comm group", __func__, group.c_str());
        return nullptr;
    }
    return g_hcclComm;
}

ReadWriteLockBase& AicpuIndopProcess::AicpuGetCommMutex()
{
    return g_commAicpuInfo.commAicpuMgrMapMutex;
}

HcclResult AicpuIndopProcess::AicpuIndOpChannelInit(HcclChannelUrmaRes *commParam)
{
    CHK_PTR_NULL(commParam);

    HCCL_INFO("[AicpuIndopProcess][%s] commParam->channelList[%p], commParam->listNum[%u], commParam->uniqueIdAddr[%p], "
        "commParam->uniqueIdSize[%u]", __func__, commParam->channelList, commParam->listNum, commParam->uniqueIdAddr,
        commParam->uniqueIdSize);

    std::string group = commParam->hcomId;
    CollCommAicpuMgr *collCommAicpuMgr = AicpuIndopProcess::AicpuGetCommMgrbyGroup(group);
    CHK_PRT_RET(collCommAicpuMgr == nullptr, HCCL_ERROR("%s collCommAicpuMgr is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);

    HcclResult ret = collCommAicpuMgr->AllocChannelResource(commParam);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuIndopProcess][AicpuIndOpChannelInit]errNo[0x%016llx] Failed to init channels group[%s]",
        HCCL_ERROR_CODE(ret), group.c_str()), ret);

    AicpuReleaseCommMgrbyGroup(group);
    HCCL_INFO("[AicpuIndopProcess][%s] aicpuTask End.", __func__);

    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::AicpuIndOpChannelUpdate(HcclChannelUrmaRes *commParam)
{
    CHK_PTR_NULL(commParam);
    HCCL_INFO("[AicpuIndopProcess][%s] commParam->channelList[%p], commParam->listNum[%u], commParam->uniqueIdAddr[%p], "
        "commParam->uniqueIdSize[%u]", __func__, commParam->channelList, commParam->listNum, commParam->uniqueIdAddr,
        commParam->uniqueIdSize);

    std::string group = commParam->hcomId;
    CollCommAicpuMgr *collCommAicpuMgr = AicpuIndopProcess::AicpuGetCommMgrbyGroup(group);
    CHK_PRT_RET(collCommAicpuMgr == nullptr, HCCL_ERROR("%s collCommAicpuMgr is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);

    HcclResult ret = collCommAicpuMgr->UpdateChannelResource(commParam);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuIndopProcess][UpdateChannelResource]errNo[0x%016llx] Failed to update channels group[%s]",
        HCCL_ERROR_CODE(ret), group.c_str()), ret);

    AicpuReleaseCommMgrbyGroup(group);
    HCCL_INFO("[AicpuIndopProcess][%s] aicpuTask End.", __func__);

    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::AicpuIndOpNotifyInit(NotifyMgrAicpuParam *param)
{
    CHK_PTR_NULL(param);

    std::string group = param->hcomId;
    HCCL_INFO("[%s]group[%s]", __func__, group.c_str());
    CollCommAicpuMgr *collCommAicpuMgr = AicpuIndopProcess::AicpuGetCommMgrbyGroup(group);
    CHK_PRT_RET(collCommAicpuMgr == nullptr, HCCL_ERROR("%s collCommAicpuMgr is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);

    HcclResult ret = HCCL_E_INTERNAL;
    if (param->freeFlag) {
        ret = collCommAicpuMgr->NotifyFree(param);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AicpuIndopProcess][%s]errNo[0x%016llx] Failed to free notifys group[%s]",
            __func__, HCCL_ERROR_CODE(ret), group.c_str()), ret);
    } else {
        ret = collCommAicpuMgr->NotifyAlloc(param);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AicpuIndopProcess][%s]errNo[0x%016llx] Failed to alloc notifys group[%s]",
            __func__, HCCL_ERROR_CODE(ret), group.c_str()), ret);
    }

    HCCL_INFO("[AicpuIndopProcess][%s] comm identifier[%s], notify op[%u] success, num[%u]",
        __func__, group.c_str(), param->freeFlag, param->notifyNum);
    AicpuReleaseCommMgrbyGroup(group);
    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::AicpuGetCommAll(std::vector<std::pair<std::string, CollCommAicpuMgr *>> &aicpuCommInfo)
{
    for (auto &kv : g_commAicpuInfo.commMgrMap) {
        aicpuCommInfo.push_back({kv.first, kv.second.get()});
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::AicpuDestroyCommbyGroup(const std::string &group)
{
    ReadWriteLock rwlock(g_commAicpuInfo.commAicpuMgrMapMutex);
    rwlock.writeLock();
    auto iter = g_commAicpuInfo.commMgrMap.find(group);
    if (iter == g_commAicpuInfo.commMgrMap.end()) {
        HCCL_ERROR("[AicpuIndopProcess][%s]group[%s] is not exist", __func__, group.c_str());
        return HCCL_E_PARA;
    }

    CollCommAicpu* aicpuComm = iter->second->GetCollCommAicpu();
    CHK_PTR_NULL(aicpuComm);
    aicpuComm->SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_INVALID);

    if (iter->second->IsUsed() == true) {
        HCCL_RUN_WARNING("[AicpuIndopProcess][%s]comm group [%s] has been used, skip erase", __func__, group.c_str());
        return HCCL_SUCCESS;
    }

    g_commAicpuInfo.commMgrMap.erase(group);
    HCCL_RUN_INFO("[AicpuIndopProcess][%s]Destroy comm group [%s] success.", __func__, group.c_str());
    rwlock.writeUnlock();
    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::AicpuDfxOpInfoInit(HcclDfxOpInfo *aicpuDfxInfo, const std::string& commTag)
{
    HCCL_INFO("[%s]group[%s], algTag[%s], profiling L0[%d], L1[%d]", __func__, commTag.c_str(), aicpuDfxInfo->algTag,
        Hccl::ProfilingHandlerLite::GetInstance().GetProfL0State(),
        Hccl::ProfilingHandlerLite::GetInstance().GetProfL1State());

    // 获取device侧的通信域
    CHK_PRT_RET(g_hcclComm == nullptr, HCCL_ERROR("%s g_hcclComm is null, commTag[%s]", __func__, commTag.c_str()), HCCL_E_PTR);
    CollCommAicpu* collComm = g_hcclComm->GetCollCommAicpu();
    CHK_PTR_NULL(collComm);

    // HcclDfxOpInfo 转为DfxOpInfo
    std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfoOnce = ConvertToDfxOpInfo(*aicpuDfxInfo);
    dfxOpInfoOnce->opIndex_ = collComm->UpdateIndex();
    dfxOpInfoOnce->comm_ = reinterpret_cast<void *>(collComm);
    dfxOpInfoOnce->isIndop_ = true;
    dfxOpInfoOnce->rankSize_ = collComm->GetTopoInfo().userRankSize;
    dfxOpInfoOnce->op_.myRank = static_cast<Hccl::RankId>(collComm->GetTopoInfo().userRank);

    // 注册
    HcclCommDfxLite* hcclCommDfxLite = collComm->GetHcclCommDfxLite();
    CHK_PTR_NULL(hcclCommDfxLite);
    CHK_RET(hcclCommDfxLite->SetCurrDfxOpInfo(dfxOpInfoOnce));
    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::ProfilingReportDeviceOp()
{
    // 获取device侧的通信域
    CHK_PTR_NULL(g_hcclComm);
    CollCommAicpu* collCommAicpu = g_hcclComm->GetCollCommAicpu();
    CHK_PTR_NULL(collCommAicpu);
    // 注册
    HcclCommDfxLite* hcclCommDfxLite = collCommAicpu->GetHcclCommDfxLite();
    Hccl::MirrorTaskManagerLite* mirrorTaskMgrLite = hcclCommDfxLite->GetMirrorTaskManagerLite();
    auto currDfxOpInfo = mirrorTaskMgrLite->GetCurrDfxOpInfo();
    if (currDfxOpInfo == nullptr) {
        HCCL_WARNING("[%s] no op info registered , skip ProfilingReportDeviceOp.", __func__);
        return HCCL_SUCCESS;
    }

    CHK_RET(hcclCommDfxLite->ReportAllTasks());
    EXCEPTION_CATCH(Hccl::ProfilingHandlerLite::GetInstance().ReportHcclOpInfo(*currDfxOpInfo),
        return HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AicpuIndopProcess::UpdateTask(const std::string &group)
{
    CHK_PTR_NULL(g_hcclComm);
    CollCommAicpu* collCommAicpu = g_hcclComm->GetCollCommAicpu();
    CHK_PTR_NULL(collCommAicpu);
    HcclCommDfxLite* hcclCommDfxLite = collCommAicpu->GetHcclCommDfxLite();
    CHK_RET(hcclCommDfxLite->UpdateProfStat());
    return HCCL_SUCCESS;
}