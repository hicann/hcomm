/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_INSTANCE_MGR_H
#define HCOMM_CCU_INSTANCE_MGR_H

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "ccu_instance.h"
#include "ccu_res_desc_mgr.h"

namespace hcomm {

class CcuInstanceMgr {
public:
    static CcuInstanceMgr& GetInstance(const int32_t deviceLogicId);

    CcuResult Init();
    CcuResult Deinit();

    // 基于实例类型创建实例
    CcuResult CreateByInsType(const CcuInstanceType insType, CcuInsHandle& insHandle);
    // 基于资源描述符数组创建实例
    CcuResult CreateByResDescs(const CcuResDesc* descs[], uint32_t descNum, CcuInsHandle& insHandle);
    // 使用当前 Device 上所有已使能 ioDie 的全部资源创建实例
    CcuResult CreateByAllRes(CcuInsHandle& insHandle);
    // 查询实例持有的资源描述符
    CcuResult QueryInsResDesc(CcuInsHandle& ccuInsHandle, uint8_t dieId, HcommCcuResDescHandle& resDesc);
    CcuInstance* Get(CcuInsHandle insHandle) const;
    CcuResult Destroy(CcuInsHandle insHandle);
    CcuResDescMgr& GetResDescMgr();

private:
    explicit CcuInstanceMgr() = default;
    ~CcuInstanceMgr();

    CcuInstanceMgr(const CcuInstanceMgr& that) = delete;
    CcuInstanceMgr& operator=(const CcuInstanceMgr& that) = delete;

private:
    bool initializedFlag_{false};
    int32_t devLogicId_{-1};
    CcuInsHandle instanceId_{0};
    mutable std::shared_timed_mutex insMapMutex_;
    std::unordered_map<CcuInsHandle, std::unique_ptr<CcuInstance>> insMap_{};
    CcuResDescMgr resDescMgr_;
};
}; // namespace hcomm

#endif // HCOMM_CCU_INSTANCE_MGR_H
