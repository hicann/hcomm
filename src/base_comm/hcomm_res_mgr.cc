/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_res_mgr.h"

#include "adapter_rts_common.h"
#include "hccl_common.h"

// orion 通用平台层单例
#include "hccp_hdc_manager.h"
#include "hccp_peer_manager.h"
#include "hccp_tlv_hdc_manager.h"
#include "rdma_handle_manager.h"
#include "inner_net_dev_manager.h"
#include "socket_handle_manager.h"
#include "host_socket_handle_manager.h"
#include "tp_manager.h"
#include "endpoint_monitor.h"
// legacy ccu单例
#include "ccu_component.h"
#include "../../../legacy/ascend950/unified_platform/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "../../../legacy/ascend950/unified_platform/ccu/ccu_context/ccu_context_mgr_imp.h"
// 开源开放 ccu单例
#include "hccp_tlv_hdc_mgr.h"
#include "tp_mgr.h"
#include "ccu_comp.h"
#include "resources/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "ccu_kernel_mgr.h"
#include "ccu_instance_mgr.h"
#include "../endpoint_pairs/sockets/socket_process.h"
#include "dpu_notify/dpu_notify_manager.h"
#include "server_socket_mgr.h"
#include "server_socket_manager.h"
#include "adapter_rts_common.h"

namespace hcomm {

HcommResMgr& HcommResMgr::GetInstance(const uint32_t devicePhyId)
{
    static std::array<bool, MAX_MODULE_DEVICE_NUM + 1> isInitialized{false};

    uint32_t devPhyId = devicePhyId;
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[HcommResMgr][%s] use the backup device, devPhyId[%u] should be "
            "less than %u.", __func__, devPhyId, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }
    if (!isInitialized[devPhyId]) {
        // 临时方案：只声明单例对象做生命周期控制，不执行业务动作
        // 未来需要将各种单例转为该数据结构的成员变量
        // devicePhyId 目前不影响流程，只是触发静态对象声明
        DpuNotifyManager::GetInstance();
        Hccl::HccpHdcManager::GetInstance();
        Hccl::HccpPeerManager::GetInstance();
        Hccl::HccpTlvHdcManager::GetInstance();
        Hccl::RdmaHandleManager::GetInstance();
        Hccl::InnerNetDevManager::GetInstance();
        Hccl::SocketHandleManager::GetInstance();
        Hccl::HostSocketHandleManager::GetInstance();
        SocketMgr::GetInstance(devicePhyId);
        Hccl::TpManager::GetInstance(devicePhyId);
        EndpointMonitor::GetInstance(devicePhyId);

        Hccl::CcuComponent::GetInstance(devicePhyId);
        Hccl::CcuResBatchAllocator::GetInstance(devicePhyId);
        Hccl::CtxMgrImp::GetInstance(devicePhyId);

        // 开源开放架构下CCU模式新增类型单例，当前混跑时不使用
        HccpTlvHdcMgr::GetInstance(devicePhyId);
        TpMgr::GetInstance(devicePhyId);
        CcuComponent::GetInstance(devicePhyId);
        CcuResBatchAllocator::GetInstance(devicePhyId);
        CcuKernelMgr::GetInstance(devicePhyId);
        CcuInstanceMgr::GetInstance(devicePhyId);
        SocketProcess::GetInstance(devicePhyId);
    }

    static HcommResMgr hcommResMgrs[MAX_MODULE_DEVICE_NUM + 1];
    hcommResMgrs[devPhyId].devPhyId_ = devPhyId;
    isInitialized[devPhyId] = true;

    return hcommResMgrs[devPhyId];
}

HcommResMgr::HcommResMgr()
{
    // 临时方案：最小化修改不做处理
    // 未来需在构造函数中触发各类成员变量声明
}

HcommResMgr::~HcommResMgr()
{
    // 临时方案：最小化修改不做处理
    // 未来需在析构函数中主动调用各种单例销毁流程，保证销毁时序
}

static void OnDeviceResetPre(int32_t deviceId, aclrtDeviceState state, void *args)
{
    try {
        if (state != ACL_RT_DEVICE_STATE_RESET_PRE) {
            return;
        }
        HCCL_INFO("[OnDeviceResetPre] deviceId[%d] state[%d] ", deviceId, static_cast<int>(state));

        u32 devPhyId = 0;
        HcclResult ret = hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceId), devPhyId);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[OnDeviceResetPre] hrtGetDevicePhyIdByIndex failed, deviceId[%d] ret[%d]", deviceId, ret);
            return;
        }
        SocketMgr::DeInit(devPhyId);
        ServerSocketMgr::DeInit(devPhyId);
        ServerSocketManager::GetInstance().DeInit(devPhyId);
        Hccl::RdmaHandleManager::GetInstance().DeInit(devPhyId);
        Hccl::SocketHandleManager::GetInstance().DeInit(devPhyId);
        Hccl::HccpHdcManager::GetInstance().DeInit(deviceId);
    } catch (const std::exception &e) {
        HCCL_WARNING("[OnDeviceResetPre][%s] exception caught:%s", __func__, e.what());
    } catch (...) {
        HCCL_WARNING("[OnDeviceResetPre][%s] unknown exception caught", __func__);
    }
}

__attribute__((constructor)) void RegisterDeviceResetCallback()
{
    aclError ret = aclrtRegDeviceStateCallback("hcomm_res_mgr", OnDeviceResetPre, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING("[RegisterDeviceResetCallback] aclrtRegDeviceStateCallback failed, ret[%d]", ret);
        return;
    }
    HCCL_INFO("[RegisterDeviceResetCallback] aclrtRegDeviceStateCallback success");
}

} // namespace hcomm
