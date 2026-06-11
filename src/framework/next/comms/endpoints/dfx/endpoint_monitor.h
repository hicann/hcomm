/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ENDPOINT_MONITOR_H
#define ENDPOINT_MONITOR_H
#include <thread>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include "log.h"
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hccp_ctx.h"

namespace hcomm {

class EndpointMonitor {
public:
    EndpointMonitor() = default;
    EndpointMonitor(const EndpointMonitor &) = delete;
    EndpointMonitor &operator=(const EndpointMonitor &) = delete;
    EndpointMonitor(EndpointMonitor &&) = delete;
    EndpointMonitor &operator=(EndpointMonitor &&) = delete;
    ~EndpointMonitor();

    static EndpointMonitor &GetInstance(s32 deviceLogicId);
    HcclResult RegisterToEndpointMonitor(s32 deviceLogicId, EndpointHandle epHandle);
    HcclResult UnRegisterToEndpointMonitor();  
    void RemoveEpHandleFromEndpointMonitor(EndpointHandle epHandle);

private:
    HcclResult RunMonitorThread();
    void MonitorThread();
    HcclResult DeInit(s32 deviceLogicId);

    void ProcessUbAsyncEvents();
    void PrintUbAsyncEventsContext(u32 devPhyId, const struct AsyncEvent &event);

    static constexpr u32 MONITOR_INTERVAL = 50;     
    std::unique_ptr<std::thread> endpointMonitorThread_;

    std::atomic<bool> initialized_{false};
    std::atomic<bool> endpointMonitorThreadFlag_{false};
    std::mutex threadLock_;
    u32 devPhyId_{0};
    s32 deviceLogicId_{0};
    std::unordered_set<u64> epHandleSet_;

    struct AsyncEvent events_[ASYNC_EVENT_MAX_NUM];
};
} // namespace hcomm
#endif // ENDPOINT_MONITOR_H