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
#include "ns_recovery/task_abort_handler.h"
#include "cluster_monitor.h"

namespace hccl {

CollCommMgr* CollCommMgr::instance_ = nullptr;
static std::once_flag instanceFlag;

CollCommMgr* CollCommMgr::GetInstance()
{
    std::call_once(instanceFlag, [&] {
        instance_ = new CollCommMgr();
    });
    return instance_;
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
    HcclTaskAbortHandler::GetInstance().Register(collComm);
    (void)GetOrderLaunchThreadMgr(collComm->GetDeviceLogicId()).RegisterOrderLaunch(collComm->GetCommId());
}

void CollCommMgr::UnRegisteCollComm(CollComm* collComm)
{
    std::lock_guard<std::mutex> lock(mutex_);
    allCollComms_.erase(collComm->GetCommId());
    // 从通信域里面注销
    HcclTaskAbortHandler::GetInstance().UnRegister(collComm);
    (void)GetClusterMonitor(collComm->GetDeviceLogicId()).UnRegisterToClusterMonitor(collComm);
    (void)GetOrderLaunchThreadMgr(collComm->GetDeviceLogicId()).UnRegisterOrderLaunch(collComm->GetCommId());
}

std::unordered_map<std::string, CollComm*> CollCommMgr::GetAllCollComms() { return allCollComms_; }

} // namespace hccl
