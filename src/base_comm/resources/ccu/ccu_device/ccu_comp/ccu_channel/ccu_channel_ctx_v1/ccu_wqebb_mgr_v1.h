/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_WQEBB_MGR_V1_H
#define CCU_WQEBB_MGR_V1_H

#include "ccu_wqebb_mgr.h"

#include <memory>

#include "ccu_res_allocator.h"

namespace hcomm {

class CcuWqeBBMgrV1 : public CcuWqeBBMgr {
public:
    CcuWqeBBMgrV1(const int32_t devLogicId, const uint8_t dieId) : CcuWqeBBMgr(devLogicId, dieId) {}
    CcuWqeBBMgrV1() = default;
    ~CcuWqeBBMgrV1() override final = default;

    HcclResult Init() override final;
    HcclResult Alloc(const WqeBBReq& wqeBBReq, ResInfo& wqeBBInfo) override final;
    HcclResult Release(const ResInfo& wqeBBInfo) override final;

private:
    std::unique_ptr<CcuResIdAllocator> idAllocator_{nullptr};
};

}; // namespace hcomm

#endif
