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

namespace hcomm {

class HcommResMgr {
public:
    static HcommResMgr& GetInstance(const uint32_t devicePhyId);
    static void RegisterDeviceResetCallback();

private:
    HcommResMgr();
    ~HcommResMgr();
    HcommResMgr(const HcommResMgr& that) = delete;
    HcommResMgr& operator=(const HcommResMgr& that) = delete;

    uint32_t devPhyId_{0};
};

} // namespace hcomm

#endif // HCOMM_RES_MGR_H
