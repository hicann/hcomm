/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_PLF_DEBUG_CONFIG_H
#define HCCLV2_PLF_DEBUG_CONFIG_H

#include "base_config_legacy.h"

namespace Hccl {

class EnvPlfDebugConfig : public BaseConfig {
public:
    void Parse() override;
    u64 GetConfigValue() const;

private:
    u64 plfDebugConfig_ = 0;
};

} // namespace Hccl

#endif // HCCLV2_PLF_DEBUG_CONFIG_H
