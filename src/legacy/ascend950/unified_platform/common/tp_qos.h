/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TP_QOS_H
#define HCCL_TP_QOS_H

#include <cstdint>

namespace Hccl {

/// UBOE 默认 DSCP：HCCN 配置未命中 qos 档位时的回退值
constexpr uint8_t kUboeDefaultDscp = 33U;

/// 按 hcclQos 与可用 SL 档位数 numGroups 计算 SL 分组下标。
uint32_t TpQosResolveQosSlGroupIdx(const uint32_t qos, const uint32_t numGroups);

/// 从 HCCN_CFG_QOS_DSCP 配置中按 qos 解析 DSCP 值。
/// 配置项 key 为 qos_dscp_{phyId}，value 格式为 "qos:dscp,qos:dscp,..."，最多 8 对，例如 "0:33,1:65"。
bool TpQosGetDscpByQosFromHccnCfg(const uint32_t devPhyId, uint8_t qos, uint8_t& dscpOut);

} // namespace Hccl

#endif // HCCL_TP_QOS_H
