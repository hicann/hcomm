/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __COLL_COMM_AICPU_MGR_H__
#define __COLL_COMM_AICPU_MGR_H__

#include "coll_comm_aicpu.h"

class CollCommAicpuMgr {
public:
    HcclResult AcquireCollCommAicpu();
    HcclResult InitAicpuIndOp(CommAicpuParam* commAicpuParam);
    HcclResult InitThreads(ThreadMgrAicpuParam* param);
    HcclResult AllocChannelResource(HcclChannelUrmaRes* commParam);
    HcclResult UpdateChannelResource(HcclChannelUrmaRes* commParam);
    HcclResult NotifyFree(NotifyMgrAicpuParam* param);
    HcclResult NotifyAlloc(NotifyMgrAicpuParam* param);

    bool IsUsed();
    void SetUsed(bool used);
    void SetOldA5Comm(void* oldA5Comm);
    CollCommAicpu* GetCollCommAicpu() { return collCommAicpu_.get(); }

private:
    void* oldA5Comm_{nullptr};
    std::unique_ptr<CollCommAicpu> collCommAicpu_{nullptr};
    std::shared_mutex isUsedMutex_;
    bool isUsed_{false};
};

#endif // __COLL_COMM_AICPU_MGR_H__
