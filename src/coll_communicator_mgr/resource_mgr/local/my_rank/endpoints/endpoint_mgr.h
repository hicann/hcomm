/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_MGR_H
#define ENDPOINT_MGR_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "endpoint.h"
#include "../endpoint_pairs/endpoint_pair.h"
#include "hccl_mem_defs.h"

namespace hcomm {
/**
 * @note 职责：Endpoint管理器，支持不同类型的Endpoint的创建和销毁管理。
 */
class EndpointMgr {
public:
    EndpointMgr() {};
    ~EndpointMgr();

    // 获取端点
    HcclResult Get(EndpointDesc epDesc, EndpointHandle& handle);

    // 注册内存到端点
    HcclResult RegisterMemory(
        EndpointHandle epHandle, const std::vector<std::string>& memTag, const std::vector<HcclMem>& memVec,
        std::vector<MemHandle>& memHandleVec);

    // 获取所有注册的内存信息
    HcclResult GetAllRegisteredMemory(EndpointHandle epHandle, std::vector<MemHandle>& memHandleVec);

private:
    HcclResult AddMemHandle(EndpointHandle endpointHandle, const std::vector<MemHandle>& memHandleVec);
    bool IsMemExist(EndpointHandle epHandle);
    bool IsDescExist(EndpointDesc epDesc);

private:
    std::unordered_map<EndpointDesc, EndpointHandle> endpointMap_{};
    std::unordered_map<EndpointHandle, std::vector<MemHandle>> endpointMemMap_{};
    std::mutex mutex_{};
};

} // namespace hcomm

#endif // ENDPOINT_MGR_H
