/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_DPU_STREAM_MANAGER_H
#define HCCLV2_DPU_STREAM_MANAGER_H

#include <memory>
#include "stream_pub.h"

namespace hccl {

class DpuStreamManager {
public:
    explicit DpuStreamManager()
    {
        stream_ = std::make_unique<Stream>(StreamType::STREAM_TYPE_ONLINE, true);
    }

    ~DpuStreamManager() = default;

    Stream *GetStream() const
    {
        return stream_.get();
    }

private:
    std::unique_ptr<Stream> stream_{nullptr};
};

}  // namespace hccl

#endif  // HCCLV2_DPU_STREAM_MANAGER_H
