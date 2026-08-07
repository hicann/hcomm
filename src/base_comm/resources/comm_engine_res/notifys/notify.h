/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef NOTIFY_H
#define NOTIFY_H

#include <memory>

namespace hcomm {
/**
 * @note 职责：同步信号Notify的C++抽象接口，当前主要支持通信引擎kernel内的主Thread等和kernel外部Stream等之间的信号同步。
 */
class Notify {
public:
    virtual ~Notify() = default;
};
} // namespace hcomm

#endif // NOTIFY_H
