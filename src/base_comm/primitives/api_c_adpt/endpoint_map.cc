/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_map.h"
#include <unordered_map>
#include <memory>
#include <mutex>

namespace hcomm {

static std::unordered_map<EndpointHandle, std::unique_ptr<Endpoint>> g_EndpointMap;
static std::mutex g_EndpointMapMutex;

void HcommEndpointMap::AddEndpoint(EndpointHandle handle, std::unique_ptr<Endpoint> endpoint)
{
    std::lock_guard<std::mutex> lock(g_EndpointMapMutex);

    auto it = g_EndpointMap.find(handle);
    if (it != g_EndpointMap.end()) {
        it->second = std::move(endpoint);
        return;
    }

    g_EndpointMap.emplace(handle, std::move(endpoint));
    return;
}

bool HcommEndpointMap::RemoveEndpoint(EndpointHandle handle)
{
    std::lock_guard<std::mutex> lock(g_EndpointMapMutex);

    auto it = g_EndpointMap.find(handle);
    if (it != g_EndpointMap.end()) {
        g_EndpointMap.erase(it);
        return true;
    }
    return false;
}

bool HcommEndpointMap::UpdateEndpoint(EndpointHandle handle, std::unique_ptr<Endpoint> newEndpoint)
{
    std::lock_guard<std::mutex> lock(g_EndpointMapMutex);

    auto it = g_EndpointMap.find(handle);
    if (it != g_EndpointMap.end()) {
        it->second = std::move(newEndpoint);
        return true;
    }
    return false;
}

Endpoint* HcommEndpointMap::GetEndpoint(EndpointHandle handle)
{
    std::lock_guard<std::mutex> lock(g_EndpointMapMutex);

    auto it = g_EndpointMap.find(handle);
    if (it != g_EndpointMap.end()) {
        return it->second.get(); // 返回裸指针，不转移所有权
    }
    return nullptr;
}
} // namespace hcomm
