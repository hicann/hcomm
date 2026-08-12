/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>

#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "log.h"
#include "endpoint.h"
#include "endpoint_monitor.h"
#include "../hcomm_res_mgr.h"
#include "hcomm_result_defs.h"
#include "param_check_pub.h"
#include "exception_handler.h"
#include "hcom_common.h"
#include "hcomm_res_defs.h"
#include "channel_config.h"
#include "shared_jetty_mgr.h"
#include "endpoint.h"
#include "builtin_endpoint_ops.h"
#include "nic_plugin_holder.h"
#include "nic_plugin_manager.h"

using namespace hcomm;

namespace {
HcclResult ValidateEndpointDesc(const EndpointDesc* endpoint, EndpointHandle* endpointHandle)
{
    CHK_PTR_NULL(endpoint);
    CHK_PTR_NULL(endpointHandle);
    if (endpoint->loc.locType != ENDPOINT_LOC_TYPE_DEVICE && endpoint->loc.locType != ENDPOINT_LOC_TYPE_HOST) {
        HCCL_ERROR(
            "[%s] Only support END_POINT_LOCATION_DEVICE AND END_POINT_LOCATION_HOST, but "
            "endpoint->loc.locType is %d",
            __func__, endpoint->loc.locType);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterDeviceEndpointMonitorIfNeeded(const EndpointDesc* endpoint, EndpointHandle handle)
{
    if ((endpoint->loc.locType != ENDPOINT_LOC_TYPE_DEVICE)
        || ((endpoint->protocol != COMM_PROTOCOL_UBC_CTP) && (endpoint->protocol != COMM_PROTOCOL_UBC_TP))) {
        return HCCL_SUCCESS;
    }

    s32 devLogicIdSigned = HcclGetThreadDeviceId();
    CHK_PRT_RET(
        devLogicIdSigned < 0, HCCL_ERROR("[%s] HcclGetThreadDeviceId failed, ret[%d]", __func__, devLogicIdSigned),
        HCCL_E_INTERNAL);
    EndpointMonitor::GetInstance(devLogicIdSigned).RegisterToEndpointMonitor(devLogicIdSigned, handle);
    return HCCL_SUCCESS;
}

HcommResult CreatePluginEndpointHolder(
    const EndpointDesc* endpoint, const NicPluginEntry* pluginEntry, EndpointHandle* endpointHandle)
{
    CHK_PTR_NULL(endpoint);
    CHK_PTR_NULL(pluginEntry);
    CHK_PTR_NULL(endpointHandle);
    void* pluginCtx = nullptr;
    HcommNicEndpointOps* pluginOps = nullptr;
    HcommResult ret = HCCL_SUCCESS;
    ret = static_cast<HcommResult>(pluginEntry->createEndpoint(endpoint, &pluginCtx, &pluginOps));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[NicPlugin][%s] plugin createEndpoint failed, ret[%d], protocol[%d].", __func__, ret, endpoint->protocol),
        ret);

    CHK_PRT_RET(
        !ValidateEndpointOps(pluginOps),
        HCCL_ERROR(
            "[NicPlugin][%s] plugin endpoint ops validation failed, protocol[%d].", __func__, endpoint->protocol),
        HCCL_E_PARA);

    // 确保HcommNicEndpointOps各接口非空实现，后续调用处无需校验
    HcommNicEndpointOps* pluginHolderOps = nullptr;
    ret = FillDefaultEndpointOps(pluginOps, &pluginHolderOps);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[NicPlugin][%s] FillDefaultEndpointOps failed, ret[%d], protocol[%d].", __func__, ret, endpoint->protocol),
        ret);

    ret = static_cast<HcommResult>(pluginHolderOps->init(pluginCtx));
    if (ret != HCCL_SUCCESS) {
        int32_t destroyRet = pluginHolderOps->destroy(pluginCtx);
        if (destroyRet != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] plugin endpoint destroy failed after init failure, ret[%d].", __func__, destroyRet);
        }
        delete pluginHolderOps;
        HCCL_ERROR("[NicPlugin][%s] plugin endpoint init failed, ret[%d].", __func__, ret);
        return ret;
    }

    auto holder = std::make_unique<PluginEndpointHolder>(*endpoint, pluginEntry);
    holder->SetNicEndpointCtx(pluginHolderOps, pluginCtx);
    const EndpointHandle handle = reinterpret_cast<EndpointHandle>(holder.get());
    EXCEPTION_CATCH(GetEndpointMap().AddEndpoint(handle, std::move(holder)), return HCCL_E_INTERNAL);
    *endpointHandle = handle;
    HCCL_INFO(
        "[NicPlugin][%s] plugin endpoint created, protocol[%d], handle[%p].", __func__, endpoint->protocol, handle);
    return HCCL_SUCCESS;
}

HcclResult CreateBuiltinEndpoint(const EndpointDesc* endpoint, EndpointHandle* endpointHandle)
{
    CHK_RET(RefreshEndpointContext(*endpoint));
    std::unique_ptr<Endpoint> endpointPtr = nullptr;
    HcclResult ret = Endpoint::CreateEndpoint(*endpoint, endpointPtr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call Endpoint::CreateEndpoint failed");
        return ret;
    }
    CHK_PTR_NULL(endpointPtr);
    ret = endpointPtr->Init();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call endpointPtr->Init failed");
        return ret;
    }

    endpointPtr->SetNicEndpointCtx(&g_BuiltinEndpointOps, endpointPtr.get());

    const EndpointHandle handle = reinterpret_cast<EndpointHandle>(endpointPtr.get());
    CHK_PTR_NULL(handle);
    EXCEPTION_CATCH(GetEndpointMap().AddEndpoint(handle, std::move(endpointPtr)), return HCCL_E_INTERNAL);
    *endpointHandle = handle;
    CHK_RET(RegisterDeviceEndpointMonitorIfNeeded(endpoint, handle));
    HCCL_INFO(
        "[%s] endpointDesc.protocol [%d] and endpointDesc.loc.locType [%d] create endpointHandle [%p] done.", __func__,
        endpoint->protocol, endpoint->loc.locType, handle);
    return HCCL_SUCCESS;
}
} // namespace

HcommResult HcommEndpointGet(EndpointHandle endpointHandle, void** endpoint) // 根据endpointHandle返回Endpoint对象指针
{
    CHK_PTR_NULL(endpoint);

    auto it = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        it == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);

    *endpoint = static_cast<void*>(it);
    HCCL_INFO(
        "[%s] START. endpointHandle[%p] endpoint[%p].", __func__, static_cast<void*>(endpointHandle),
        static_cast<void*>(endpoint));
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle)
{
    EXCEPTION_HANDLE_BEGIN
    CHK_RET(ValidateEndpointDesc(endpoint, endpointHandle));
    if (endpoint->loc.locType == ENDPOINT_LOC_TYPE_HOST) {
        const NicPluginEntry* pluginEntry = FindHostNicPlugin(endpoint->protocol);
        if (pluginEntry != nullptr) {
            return CreatePluginEndpointHolder(endpoint, pluginEntry, endpointHandle);
        }
    }
    (void)HcommResMgrInit();
    CHK_RET(CreateBuiltinEndpoint(endpoint, endpointHandle));
    HcommResMgr::RegisterDeviceResetCallback();
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle)
{
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    if (endpoint != nullptr && endpoint->GetNicOps() != nullptr && endpoint->GetNicOps() != &g_BuiltinEndpointOps) {
        HCCL_INFO("[NicPlugin][%s] destroy plugin endpoint.", __func__);
        auto ret = GetEndpointMap().RemoveEndpoint(endpointHandle);
        CHK_PRT_RET(
            ret == false, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle),
            HCCL_E_NOT_FOUND);
        return HCCL_SUCCESS;
    }
    (void)HcommResMgrInit();
    // 需校验共享 jetty channel 是否已全部销毁，否则残留 channel 持有的 jetty 引用会在 endpoint 销毁后成为悬空引用。
    HcclResult jettyRet = hcomm::SharedJettyMgr::GetInstance().CheckEndpointDestroy(endpointHandle);
    CHK_PRT_RET(
        jettyRet != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] cannot destroy endpointHandle[0x%llx], shared jetty channels still exist.", __func__, endpointHandle),
        jettyRet);
    if (endpoint != nullptr) {
        CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    }
    s32 devLogicIdSigned = HcclGetThreadDeviceId();
    CHK_PRT_RET(
        devLogicIdSigned < 0, HCCL_ERROR("[%s] HcclGetThreadDeviceId failed, ret[%d]", __func__, devLogicIdSigned),
        HCCL_E_INTERNAL);
    EndpointMonitor::GetInstance(devLogicIdSigned).RemoveEpHandleFromEndpointMonitor(endpointHandle);
    auto ret = GetEndpointMap().RemoveEndpoint(endpointHandle);
    CHK_PRT_RET(
        ret == false, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointStartListen(EndpointHandle endpointHandle, uint32_t port, HcommEndpointListenConfig* config)
{
    (void)config;
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->ServerSocketListen(port));
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointStopListen(EndpointHandle endpointHandle, uint32_t port)
{
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->ServerSocketStopListen(port));
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port)
{
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    return static_cast<HcclResult>(endpoint->GetNicOps()->getListenPort(endpoint->GetNicCtx(), port));
}

HcommResult
HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc* endpointDesc, bool* value)
{
    CHK_PTR_NULL(endpointDesc);
    CHK_PTR_NULL(value);
    (void)HcommResMgrInit();

    return static_cast<HcommResult>(Endpoint::CheckFeature(*endpointDesc, featureType, *value));
}
