/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CPU_ROCE_CHANNEL_H
#define CPU_ROCE_CHANNEL_H

#include <memory>
#include <vector>

namespace hcomm {
/**
 * @note 职责：Channel的CPU通信引擎、RoCE协议的类派生，CPU直调RoCE
 */
class CpuRoceChannel : public Channel {
public:
    CpuRoceChannel();
    virtual ~CpuRoceChannel() = default;
};
} // namespace hcomm
#endif // CPU_ROCE_CHANNEL_H
