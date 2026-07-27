/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CONFIG_PLF_LOG_H
#define CONFIG_PLF_LOG_H

#include "log.h"

namespace Hccl {
constexpr u64 PLF_ALG     = 0x1ULL << 0;   // bit 0 — 预留
constexpr u64 PLF_TASK    = 0x1ULL << 1;   // bit 1 — TASK 日志
constexpr u64 PLF_RES     = 0x1ULL << 2;   // bit 2 — 预留
constexpr u64 PLF_DATA_OP = 0x1ULL << 3;   // bit 3 — DATA_OP 日志

u64 GetPlfDebugConfigValue();
void SetPlfDebugConfigValue(u64 value);
}

// config要求传入宏名字作为日志打印关键字，不可以传入其他变量或常量
#define PLF_CONFIG_INFO(config, format, ...) do {                                              \
    if (UNLIKELY(Hccl::GetPlfDebugConfigValue() & (config))) {                                  \
        const char* configName = #config;                                                         \
        LOG_FUNC(static_cast<u32>(HCCL) | RUN_LOG_MASK, HCCL_LOG_INFO,                           \
            "[%s:%d] [%u] [%s]: " format,                                                         \
            __FILE__, __LINE__, static_cast<u32>(syscall(SYS_gettid)), configName, ##__VA_ARGS__); \
    } else {                                                                                     \
        HCCL_INFO(format, ##__VA_ARGS__);                                                        \
    }                                                                                           \
} while(0)

#define PLF_CONFIG_DEBUG(config, format, ...) do {                                             \
    if (UNLIKELY(Hccl::GetPlfDebugConfigValue() & (config))) {                                  \
        const char* configName = #config;                                                         \
        LOG_FUNC(static_cast<u32>(HCCL) | RUN_LOG_MASK, HCCL_LOG_INFO,                           \
            "[%s:%d] [%u] [%s]: " format,                                                         \
            __FILE__, __LINE__, static_cast<u32>(syscall(SYS_gettid)), configName, ##__VA_ARGS__);                  \
    } else {                                                                                     \
        HCCL_DEBUG(format, ##__VA_ARGS__);                                                       \
    }                                                                                           \
} while(0)

#endif  // CONFIG_PLF_LOG_H
