/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_rtp_endpoint.h"
#include "endpoint_mgr.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "urma_mem.h"
#include "adapter_rts_common.h"

namespace hcomm {

UbRtpEndpoint::UbRtpEndpoint(const EndpointDesc& endpointDesc) : UboeUbRtpEndpointHelper(endpointDesc) {}

HcclResult UbRtpEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        HCCL_ERROR(
            "[%s] UB_RTP protocol only supports DEV_TYPE_950/960, current deviceType=%d", __func__,
            static_cast<int>(deviceType));
        return HCCL_E_NOT_SUPPORT;
    }

    // UB_RTP 直接从 EID type CommAddr 获取地址，不做 IP→EID 转换
    Hccl::IpAddress eidAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, eidAddr));

    s32 deviceLogicId;
    u32 devPhyId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devPhyId));
    endpointDesc_.loc.device.devPhyId = devPhyId;

    Hccl::HccpHdcManager::GetInstance().Init(deviceLogicId);
    // UB_RTP 直接用 EID 地址获取 rdmaHandle，不做 IP→EID 查询
    auto& rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    EXCEPTION_CATCH(
        ctxHandle_ = static_cast<void*>(rdmaHandleMgr.GetByIp(endpointDesc_.loc.device.devPhyId, eidAddr)),
        return HCCL_E_PARA);
    CHK_PTR_NULL(ctxHandle_);
    HCCL_INFO(
        "%s success, devPhyId[%u], eidAddr[%s], ctxHandle[%p]", __func__, devPhyId, eidAddr.Describe().c_str(),
        ctxHandle_);

    EXCEPTION_CATCH(regedMemMgr_ = std::make_unique<UbRegedMemMgr>(), return HCCL_E_INTERNAL);
    regedMemMgr_->rdmaHandle_ = ctxHandle_;

    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm
