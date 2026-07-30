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
#ifdef ENABLE_EXPERIMENTAL
#include "nic_plugin_dispatcher.h"
#endif

using namespace hcomm;

namespace {
HcclResult ValidateEndpointDesc(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
    CHK_PTR_NULL(endpoint);
    CHK_PTR_NULL(endpointHandle);
    if (endpoint->loc.locType != ENDPOINT_LOC_TYPE_DEVICE && endpoint->loc.locType != ENDPOINT_LOC_TYPE_HOST) {
        HCCL_ERROR("[%s] Only support END_POINT_LOCATION_DEVICE AND END_POINT_LOCATION_HOST, but "
                   "endpoint->loc.locType is %d",
            __func__, endpoint->loc.locType);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterDeviceEndpointMonitorIfNeeded(const EndpointDesc *endpoint, EndpointHandle handle)
{
    if ((endpoint->loc.locType != ENDPOINT_LOC_TYPE_DEVICE)
        || ((endpoint->protocol != COMM_PROTOCOL_UBC_CTP) && (endpoint->protocol != COMM_PROTOCOL_UBC_TP))) {
        return HCCL_SUCCESS;
    }

    s32 devLogicIdSigned = HcclGetThreadDeviceId();
    CHK_PRT_RET(devLogicIdSigned < 0,
        HCCL_ERROR("[%s] HcclGetThreadDeviceId failed, ret[%d]", __func__, devLogicIdSigned), HCCL_E_INTERNAL);
    EndpointMonitor::GetInstance(devLogicIdSigned).RegisterToEndpointMonitor(devLogicIdSigned, handle);
    return HCCL_SUCCESS;
}

HcclResult CreateBuiltinEndpoint(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
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

    const EndpointHandle handle = reinterpret_cast<EndpointHandle>(endpointPtr.get());
    CHK_PTR_NULL(handle);
    EXCEPTION_CATCH(GetEndpointMap().AddEndpoint(handle, std::move(endpointPtr)), return HCCL_E_INTERNAL);
    *endpointHandle = handle;
    CHK_RET(RegisterDeviceEndpointMonitorIfNeeded(endpoint, handle));
    HCCL_INFO("[%s] endpointDesc.protocol [%d] and endpointDesc.loc.locType [%d] create endpointHandle [%p] done.",
        __func__, endpoint->protocol, endpoint->loc.locType, handle);
    return HCCL_SUCCESS;
}
} // namespace

HcommResult HcommEndpointGet(EndpointHandle endpointHandle, void **endpoint) // 根据endpointHandle返回Endpoint对象指针
{
    CHK_PTR_NULL(endpoint);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginEndpointGet(endpointHandle, endpoint, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto it = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(it == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);

    *endpoint = static_cast<void *>(it);
    HCCL_INFO("[%s] START. endpointHandle[%p] endpoint[%p].", __func__, static_cast<void*>(endpointHandle), static_cast<void*>(endpoint));
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
    EXCEPTION_HANDLE_BEGIN
    (void) HcommResMgrInit();
    CHK_RET(ValidateEndpointDesc(endpoint, endpointHandle));
#ifdef ENABLE_EXPERIMENTAL
    bool pluginHandled = false;
    CHK_RET(static_cast<HcclResult>(PluginEndpointCreate(endpoint, endpointHandle, pluginHandled)));
    if (pluginHandled) {
        HCCL_INFO("[NicPluginDebug][%s] plugin endpoint created, protocol[%d], handle[%p].", __func__,
            endpoint->protocol, *endpointHandle);
        return HCCL_SUCCESS;
    }
#endif
    CHK_RET(CreateBuiltinEndpoint(endpoint, endpointHandle));
    HcommResMgr::RegisterDeviceResetCallback();
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle)
{
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginEndpointDestroy(endpointHandle, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    if (endpoint != nullptr) {
        CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    }
    s32 devLogicIdSigned = HcclGetThreadDeviceId();
    CHK_PRT_RET(devLogicIdSigned < 0,
        HCCL_ERROR("[%s] HcclGetThreadDeviceId failed, ret[%d]", __func__, devLogicIdSigned), HCCL_E_INTERNAL);
    EndpointMonitor::GetInstance(devLogicIdSigned).RemoveEpHandleFromEndpointMonitor(endpointHandle);
    auto ret = GetEndpointMap().RemoveEndpoint(endpointHandle);
    CHK_PRT_RET(ret == false, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointStartListen(EndpointHandle endpointHandle, uint32_t port, HcommEndpointListenConfig *config)
{
    (void)config;
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->ServerSocketListen(port));
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointStopListen(EndpointHandle endpointHandle, uint32_t port)
{
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->ServerSocketStopListen(port));
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t *port)
{
    CHK_PTR_NULL(port);
    (void)HcommResMgrInit();
#ifdef ENABLE_EXPERIMENTAL
    if (IsPluginEndpoint(endpointHandle)) {
        return HCCL_E_NOT_SUPPORT;
    }
#endif

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr,
        HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    return endpoint->ServerSocketGetListenPort(port);
}

HcommResult HcommEndpointCheckFeature(
    HcommEndpointFeatureType featureType, const EndpointDesc *endpointDesc, bool *value)
{
    CHK_PTR_NULL(endpointDesc);
    CHK_PTR_NULL(value);
    (void)HcommResMgrInit();

    return static_cast<HcommResult>(Endpoint::CheckFeature(*endpointDesc, featureType, *value));
}
