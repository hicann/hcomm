/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "endpoint_mgr.h"
#include <algorithm>
#include "hcomm_c_adpt.h"

namespace hcomm {

EndpointMgr::~EndpointMgr()
{
    for (const auto& kv : endpointMemMap_) {
        const EndpointHandle& endpointHandle = kv.first;
        const std::vector<MemHandle>& memHandleVec = kv.second;

        for (auto menHandle : memHandleVec) {
            (void)HcommMemUnreg(endpointHandle, menHandle);
        }
    }

    for (const auto& kv : endpointMap_) {
        const EndpointHandle& endpointHandle = kv.second;
        (void)HcommEndpointDestroy(endpointHandle);
    }
}

HcclResult EndpointMgr::Get(EndpointDesc epDesc, EndpointHandle& handle)
{
    auto iterPtr = endpointMap_.find(epDesc);
    if (iterPtr != endpointMap_.end()) {
        handle = iterPtr->second;
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[EndpointMgr::Get] create Endpoint");
    CHK_RET(static_cast<HcclResult>(HcommEndpointCreate(&epDesc, &handle)));

    endpointMap_.emplace(epDesc, handle);
    return HCCL_SUCCESS;
}

HcclResult EndpointMgr::RegisterMemory(
    EndpointHandle epHandle, const std::vector<std::string>& memTag, const std::vector<HcclMem>& memVec,
    std::vector<MemHandle>& memHandleVec)
{
    memHandleVec.clear();
    uint32_t index = 0;
    for (const auto& mem : memVec) {
        MemHandle memHandle = nullptr;
        CommMem commMem{static_cast<CommMemType>(mem.type), mem.addr, mem.size};
        HcclResult ret = static_cast<HcclResult>(HcommMemReg(epHandle, memTag[index].c_str(), &commMem, &memHandle));
        if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
            HCCL_ERROR("[%s]call trace: hcclRet -> %d", __FUNCTION__, ret);
            return ret;
        }
        CHK_PTR_NULL(memHandle);
        memHandleVec.push_back(memHandle);
        index++;
        if (ret == HCCL_E_AGAIN) {
            HCCL_WARNING("This mem has already been registered, addr=%p, size=%llu", mem.addr, mem.size);
        }
    }
    CHK_RET(AddMemHandle(epHandle, memHandleVec));
    return HCCL_SUCCESS;
}

HcclResult EndpointMgr::AddMemHandle(EndpointHandle epHandle, const std::vector<MemHandle>& memHandleVec)
{
    if (memHandleVec.empty()) {
        return HCCL_SUCCESS;
    }

    if (IsMemExist(epHandle)) {
        auto& existMemHandleVec = endpointMemMap_.at(epHandle);
        existMemHandleVec.insert(existMemHandleVec.end(), memHandleVec.begin(), memHandleVec.end());
        return HCCL_SUCCESS;
    }

    endpointMemMap_.emplace(epHandle, std::move(memHandleVec));
    return HCCL_SUCCESS;
}

bool EndpointMgr::IsMemExist(EndpointHandle epHandle)
{
    return endpointMemMap_.find(epHandle) != endpointMemMap_.end();
}

bool EndpointMgr::IsDescExist(EndpointDesc epDesc) { return endpointMap_.find(epDesc) != endpointMap_.end(); }

HcclResult EndpointMgr::GetAllRegisteredMemory(EndpointHandle epHandle, std::vector<MemHandle>& memHandleVec)
{
    if (!IsMemExist(epHandle)) {
        HCCL_ERROR("EndpointMgr GetAllRegisteredMemory Fail");
        return HCCL_E_MEMORY;
    }
    memHandleVec = endpointMemMap_.at(epHandle);
    return HCCL_SUCCESS;
}

} // namespace hcomm
