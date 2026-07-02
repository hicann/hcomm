/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "hccl_device_pub.h"
#include "hccl_kernel_executor.h"
#include "sim_log.h"
#include "db_sim_op_db_ops.h"

using namespace HcclSim;

namespace sim {
    extern uint32_t g_currOpDetailId;
}

int main(int argc, char *argv[])
{
    LogConfig config = LoadLogConfig("device_aarch64");
    InitLogger(config);

    RegisterSignalHandler();
    setvbuf(stdout, nullptr, _IOLBF, 0);
    HCCL_VM_INFO("main process start.");
    
    uint32_t rankId = static_cast<uint32_t>(std::stoi(argv[1]));
    SetCurRankId(rankId);
    HCCL_VM_INFO("rankId[{}] process start...", rankId);

    HcclAicpuData *aicpuData = GetHcclAicpuDataShmPtr();
    if (aicpuData == nullptr) {
        HCCL_VM_ERROR("get HcclAicpuData shm ptr failed.");
        return -1;
    }

    aicpuData->task[rankId].devState = DEVICE_WAIT;
    while (true) {
        if (aicpuData->task[rankId].devState == DEVICE_WAIT) {
            sleep(1);
            continue;
        }

        std::atomic_thread_fence(std::memory_order_acquire);
        std::string kernelName(aicpuData->task[rankId].kernelName);
        if (kernelName == "HcclLaunchAicpuKernel") {
            uint32_t opDetailId = 0;
            int ret = sim::QueryNewestOpDeatailIdByPid(getppid(), opDetailId);
            if (ret != 0) {
                HCCL_VM_ERROR("rankId[{}] QueryNewestOpDeatailIdByPid failed.", rankId);
            }
            sim::g_currOpDetailId = opDetailId;
        }

        ExecuteAicpuKernel(rankId, kernelName, aicpuData->task[rankId].args);
        aicpuData->task[rankId].devState = DEVICE_WAIT;
    }

    _exit(0);
}
