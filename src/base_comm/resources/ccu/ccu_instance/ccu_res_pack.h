/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_PACK_H
#define CCU_RES_PACK_H

#include "ccu_types.h"
#include "ccu_res_desc.h"
#include "ccu_device_pub.h"

namespace hcomm {

class CcuResPack {
public:
    explicit CcuResPack() {};
    ~CcuResPack();
    // 基于实例类型初始化
    CcuResult InitByInsType(const CcuInstanceType insType);
    // 基于资源描述符数组初始化（资源数量由 resDesc 驱动，与 Init 的 insType 路径区分）
    CcuResult InitByResDescs(const CcuResDesc* descs[], uint32_t descNum);
    CcuResult Reset();

    CcuResRepository& GetCcuResRepo();

private:
    CcuResPack(const CcuResPack& that) = delete;
    CcuResPack& operator=(const CcuResPack& that) = delete;
    CcuResPack(CcuResPack&& that) = delete;
    CcuResPack& operator=(CcuResPack&& that) = delete;

    int32_t devLogicId_{0};
    CcuResHandle resHandle_{nullptr};
    CcuResRepository resRepo_{};
};

} // namespace hcomm

#endif // CCU_RES_PACK_H
