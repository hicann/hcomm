/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "plf_debug_config.h"
#include "env_config.h"
#include "config_plf_log.h"

namespace Hccl {

static u64 ParseDebugConfig(const char* envName, u64 domainMask)
{
    char* env = getenv(envName);
    if (env == nullptr) {
        return 0;
    }

    bool invert = (env[0] == '^');
    // 第一个字符是'^', 使用取反模式，用户配置的项关闭，未配置的项打开
    u64 result = invert ? domainMask : 0ULL;

    char* configDup = strdup(invert ? env + 1 : env); // 去掉'^'符号
    if (configDup == nullptr) {
        HCCL_ERROR("[%s] strdup failed, env[%s]", __func__, env);
        return result;
    }
    char* left = nullptr;
    char* subConfig = strtok_r(configDup, ",", &left); // 按逗号分割
    while (subConfig != nullptr) {
        u64 mask = 0;
        if ((domainMask & PLF_TASK) && strcasecmp(subConfig, "TASK") == 0) {
            mask = PLF_TASK;
        } else if ((domainMask & PLF_ALG) && strcasecmp(subConfig, "ALG") == 0) {
            mask = PLF_ALG;
        } else if ((domainMask & PLF_RES) && strcasecmp(subConfig, "RESOURCE") == 0) {
            mask = PLF_RES;
        } else if ((domainMask & PLF_DATA_OP) && strcasecmp(subConfig, "DATA_OP") == 0) {
            mask = PLF_DATA_OP;
        } else {
            HCCL_ERROR("%s:%s subConfig:%s is not supported", envName, env, subConfig);
            free(configDup);
            return 0;
        }
        if (invert) {
            result &= ~mask;
        } else {
            result |= mask;
        }
        subConfig = strtok_r(nullptr, ",", &left);
    }
    free(configDup);
    HCCL_RUN_INFO("[HCCL_ENV] %s set by [%s] to [0x%llx]", envName, env, result);
    return result;
}

void EnvPlfDebugConfig::Parse()
{
    plfDebugConfig_  = ParseDebugConfig("HCCL_DEBUG_CONFIG", PLF_TASK | PLF_ALG | PLF_RES);
    plfDebugConfig_ |= ParseDebugConfig("HCOMM_DEBUG_CONFIG", PLF_TASK | PLF_DATA_OP);
    HCCL_RUN_INFO("[HCCL_ENV] plfDebugConfig set to [0x%llx]", plfDebugConfig_);
}

u64 EnvPlfDebugConfig::GetConfigValue() const
{
    return plfDebugConfig_;
}

u64 GetPlfDebugConfigValue()
{
    return EnvConfig::GetInstance().GetPlfDebugConfig().GetConfigValue();
}

}  // namespace Hccl
