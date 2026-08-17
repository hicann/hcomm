/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "uboe_endpoint.h"
#include "endpoint_mgr.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "urma_mem.h"
#include "proc_reged_mem_mgr_cache.h"
#include "adapter_rts_common.h"

namespace hcomm {

UboeEndpoint::UboeEndpoint(const EndpointDesc& endpointDesc) : UboeUbRtpEndpointHelper(endpointDesc) {}

UboeEndpoint::~UboeEndpoint() noexcept { ProcRegedMemMgrCache::GetInstance().Release(cacheKey_); }

HcclResult UboeEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, ipAddr));

    s32 deviceLogicId;
    u32 devPhyId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicId, devPhyId));
    endpointDesc_.loc.device.devPhyId = devPhyId;

    Hccl::HccpHdcManager::GetInstance().Init(deviceLogicId);
    auto& rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    Hccl::IpAddress eidAddress{};
    rdmaHandleMgr.UboeIpv4ToEid(ipAddr, eidAddress, devPhyId);
    EXCEPTION_CATCH(
        ctxHandle_ = static_cast<void*>(rdmaHandleMgr.GetByIp(endpointDesc_.loc.device.devPhyId, eidAddress)),
        return HCCL_E_PARA);
    CHK_PTR_NULL(ctxHandle_);
    HCCL_INFO(
        "%s success, devPhyId[%u], eidAddress[%s], ctxHandle[%p]", __func__, devPhyId, eidAddress.Describe().c_str(),
        ctxHandle_);

    cacheKey_ = MemMgrCacheKey{devPhyId, COMM_PROTOCOL_UB_CTP, ipAddr, LocTypeToPortType(endpointDesc_.loc.locType)};
    auto& cache = ProcRegedMemMgrCache::GetInstance();
    EXCEPTION_CATCH(
        regedMemMgr_ = cache.GetOrCreate(
            cacheKey_,
            [this]() {
                auto m = std::make_shared<UbRegedMemMgr>();
                m->rdmaHandle_ = this->ctxHandle_;
                return m;
            }),
        return HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm
