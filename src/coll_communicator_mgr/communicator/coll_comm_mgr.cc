/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_comm_mgr.h"
#include "cluster_monitor.h"
#include "hcomm_c_adpt.h"
#include "hcom_common.h"

namespace hccl {

CollCommMgr& CollCommMgr::GetInstance()
{
    static CollCommMgr instance;
    return instance;
}

CollCommMgr::~CollCommMgr()
{
    HCCL_INFO("[CollCommMgr][~CollCommMgr] destruct begin.");
    for (auto& monitor : clusterMonitor_) {
        (void)monitor.DeInit();
    }
    HCCL_INFO("[CollCommMgr][~CollCommMgr] destruct end.");
}

hcomm::ClusterMonitor& CollCommMgr::GetClusterMonitor(s32 deviceLogicId)
{
    if (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[ClusterMonitor][%s]deviceLogicId[%d] >= %u, invalid", __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
        return clusterMonitor_[0];
    }
    return clusterMonitor_[deviceLogicId];
}

HcclResult CollCommMgr::TryReserveCcuMsComm(s32 deviceLogicId, const std::string& commId, bool& reserved)
{
    reserved = false;
    if (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM || commId.empty()) {
        HCCL_ERROR(
            "[%s] invalid parameter, deviceLogicId[%d], max device num[%u], commId empty[%d].", __func__, deviceLogicId,
            MAX_MODULE_DEVICE_NUM, commId.empty());
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(ccuMsCommMutex_);
    auto& owner = ccuMsCommIds_[deviceLogicId];
    if (owner.empty()) {
        owner = commId;
        reserved = true;
    }
    return HCCL_SUCCESS;
}

void CollCommMgr::ReleaseCcuMsComm(s32 deviceLogicId, const std::string& commId)
{
    if (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[%s] deviceLogicId[%d] is invalid, max device num[%u].", __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
        return;
    }

    std::lock_guard<std::mutex> lock(ccuMsCommMutex_);
    auto& owner = ccuMsCommIds_[deviceLogicId];
    if (owner == commId) {
        owner.clear();
    }
}

OrderLaunchThreadMgr& CollCommMgr::GetOrderLaunchThreadMgr(s32 deviceLogicId)
{
    if (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[CollCommMgr][%s]deviceLogicId[%d] >= %u, invalid", __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
        return orderLaunchThreadMgrs_[0];
    }
    return orderLaunchThreadMgrs_[deviceLogicId];
}

void CollCommMgr::RegisteCollComm(CollComm* collComm)
{
    std::lock_guard<std::mutex> lock(mutex_);
    allCollComms_[collComm->GetCommId()] = collComm;
    // 注册到需要的地方
    taskAbortHandler_.Register(collComm);
    (void)GetOrderLaunchThreadMgr(collComm->GetDeviceLogicId()).RegisterOrderLaunch(collComm->GetCommId());
}

void CollCommMgr::UnRegisteCollComm(CollComm* collComm)
{
    std::lock_guard<std::mutex> lock(mutex_);
    allCollComms_.erase(collComm->GetCommId());
    // 从通信域里面注销
    taskAbortHandler_.UnRegister(collComm);
    (void)GetClusterMonitor(collComm->GetDeviceLogicId()).UnRegisterToClusterMonitor(collComm);
    (void)GetOrderLaunchThreadMgr(collComm->GetDeviceLogicId()).UnRegisterOrderLaunch(collComm->GetCommId());
}

const std::unordered_map<std::string, CollComm*>& CollCommMgr::GetAllCollComms() const { return allCollComms_; }

void CollCommMgr::InitBaseCommRes(uint32_t devId) { (void)HcommResMgrInit(devId); }

HcclOpInfoCtx& CollCommMgr::LegacyGetOpHcomInfo(uint32_t devId)
{
    if (devId >= MAX_MODULE_DEVICE_NUM + 1) {
        devId = MAX_MODULE_DEVICE_NUM;
    }
    // baseCommInited_ 无需加锁：本函数在生产路径中始终由 LegacyGetHcclExistDeviceOpInfoCtx /
    // LegacyGetHcclOpInfoCtx 在 opHcomInfosMutex_ 锁内调用
    if (!baseCommInited_[devId]) {
        InitBaseCommRes(devId);
        baseCommInited_[devId] = true;
    }
    return opHcomInfos_[devId];
}

HcclOpInfoCtx& CollCommMgr::LegacyGetHcclExistDeviceOpInfoCtx(s32& devId)
{
    std::lock_guard<std::mutex> lock(opHcomInfosMutex_);
    auto& opHcomInfo = LegacyGetOpHcomInfo(devId);
    if (!opHcomInfo.isUsed) {
        HCCL_INFO("[LegacyGetHcclOpInfoCtx] Set device, use devId[%d] ", devId);
        auto& backUpOpHcomInfo = LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM);
        if (backUpOpHcomInfo.isUsed) {
            devId = MAX_MODULE_DEVICE_NUM;
            HCCL_INFO("[LegacyGetHcclOpInfoCtx] Used cover bottom devId[%d]", devId);
            return backUpOpHcomInfo;
        }
    }

    HCCL_INFO("[LegacyGetHcclExistDeviceOpInfoCtx] use devId[%d] opHcomInfos", devId);
    opHcomInfo.isUsed = true;
    return opHcomInfo;
}

HcclOpInfoCtx& CollCommMgr::LegacyGetHcclOpInfoCtx(s32& devId)
{
    if (HcclGetDeviceId() == HCCL_SUCCESS) {
        return LegacyGetHcclExistDeviceOpInfoCtx(devId);
    }

    std::lock_guard<std::mutex> lock(opHcomInfosMutex_);
    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        auto& opHcomInfo = LegacyGetOpHcomInfo(i);
        if (opHcomInfo.isUsed) {
            devId = i;
            HCCL_INFO("[LegacyGetHcclOpInfoCtx] Not set device, Used devId[%u] ", i);
            return opHcomInfo;
        }
    }

    devId = MAX_MODULE_DEVICE_NUM;
    auto& backUpOpHcomInfo = LegacyGetOpHcomInfo(devId);
    backUpOpHcomInfo.isUsed = true;
    HCCL_INFO("[LegacyGetHcclOpInfoCtx] Used cover bottom devId[%d]", devId);
    return backUpOpHcomInfo;
}

} // namespace hccl
