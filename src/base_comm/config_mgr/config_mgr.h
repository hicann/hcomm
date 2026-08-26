/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CONFIG_MGR_H
#define HCOMM_CONFIG_MGR_H

#include "base_config.h"

namespace hcomm {

/**
 * @brief 顶层环境变量聚合配置（对齐 legacy EnvConfig）。
 *
 * 持有各子配置类（如 rdmaCfg），Parse() 统一触发各子配置解析。
 * 业务侧通过 HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig() 访问
 */
class ConfigMgr {
public:
    ConfigMgr();
    ~ConfigMgr() = default;

    EnvRdmaConfig& GetRdmaConfig();

private:
    ConfigMgr(const ConfigMgr&) = delete;
    ConfigMgr& operator=(const ConfigMgr&) = delete;

    // 子配置成员
    EnvRdmaConfig rdmaCfg_;
};

} // namespace hcomm

#endif // HCOMM_CONFIG_MGR_H
