/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_MGR_H
#define HCOMM_RES_MGR_H

#include <cstdint>
#include <mutex>
#include "acl/acl_rt.h"
#include "hcomm_res_defs.h"
#include "hccl_common.h"

namespace hcomm {

class HcommResMgr {
public:
    static HcommResMgr& GetInstance(const uint32_t devicePhyId);
    static void RegisterDeviceResetCallback();

    // 进程级 kernel bin 句柄管理（不分 device，与 per-device 实例隔离）
    static HcclResult EnsureKernelBinLoaded(CommEngine engine);
    static aclrtBinHandle GetBinHandle();

private:
    HcommResMgr();
    ~HcommResMgr();
    HcommResMgr(const HcommResMgr& that) = delete;
    HcommResMgr& operator=(const HcommResMgr& that) = delete;

    uint32_t devPhyId_{0};

    static aclrtBinHandle binHandle_;
    static std::mutex binHandleMtx_;
};

} // namespace hcomm

#endif // HCOMM_RES_MGR_H
