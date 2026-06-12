/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCLV2_ENV_CONFIG_H
#define HCCLV2_ENV_CONFIG_H

#include "hcomm_res_defs.h"
#include "base_config.h"

namespace Hccl {

constexpr u32 UB_QOS_DEFAULT = HCOMM_UB_QOS_DEFAULT;

class EnvConfig {
public:
    static EnvConfig &GetInstance();

    const EnvHostNicConfig &GetHostNicConfig();

    const EnvSocketConfig &GetSocketConfig();

    const EnvRtsConfig &GetRtsConfig();

    const EnvRdmaConfig &GetRdmaConfig();

    const EnvAlgoConfig &GetAlgoConfig();

    const EnvLogConfig &GetLogConfig();

    const EnvDetourConfig &GetDetourConfig();

    EnvConfig(const EnvConfig &envConfig) = delete;

    EnvConfig &operator=(const EnvConfig &envConfig) = delete;

    void Parse();

private:
    EnvHostNicConfig       hostNicCfg;
    EnvSocketConfig        socketCfg;
    EnvRtsConfig           rtsCfg;
    EnvRdmaConfig          rdmaCfg;
    EnvAlgoConfig          algoCfg;
    EnvLogConfig           logCfg;
    EnvDetourConfig        detourCfg;
    EnvConfig();
};

} // namespace Hccl

#endif // HCCLV2_ENV_CONFIG_H