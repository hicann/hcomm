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

#include "hccl_types.h"
#include "hccp_tp.h"
#include "ip_address.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

/// PCIe 标卡 UB 互通固定 jetty priority（SL）
constexpr uint32_t kTpQosPcieStdMappedSl = 2U;

constexpr uint32_t kTpQosAttrSlAvailableBit = 17U;
constexpr uint32_t kTpQosAttrBitmapSl = (1U << 10U);
constexpr uint32_t kTpQosAttrBitmapDscp = (1U << 8U);
constexpr uint32_t kTpQosAttrDscpConfigModeBit = 18U;
/// UBOE SetTpAttr 网络属性位图：bit2~8（0x1FC，sip/dip/smac/dmac/vlan + dscp，不含 sl）；HCCP 自动 urma_get_smac/get_dmac
constexpr uint32_t kTpQosAttrBitmapUboeNetWithDscp = 0x1FCU;

using TpQosMapKey = uint32_t;

/// hcclQos → TP 列表索引 / SL / jetty priority 的策略输入（与具体 GetTpInfo 参数类型解耦）
struct TpQosPolicyInput {
    TpProtocol tpProtocol{TpProtocol::CTP};
    uint32_t qos{kRaUbGetTpInfoParamDefaultQos};
    uint32_t slLevelCount{0U};
    bool loopFirstTpLowestSl{false};
    IpAddress locIpv4Addr{};
    IpAddress rmtIpv4Addr{};
};

inline TpQosMapKey TpQosMapKeyFromQos(uint32_t qos) noexcept
{
    return static_cast<TpQosMapKey>(qos & 0xFFU);
}

inline TpQosPolicyInput TpQosPolicyInputFrom(const RaUbGetTpInfoParam &param)
{
    TpQosPolicyInput out{};
    out.tpProtocol = param.tpProtocol;
    out.qos = param.qos;
    out.slLevelCount = param.slLevelCount;
    out.loopFirstTpLowestSl = param.loopFirstTpLowestSl;
    out.locIpv4Addr = param.locIpv4Addr;
    out.rmtIpv4Addr = param.rmtIpv4Addr;
    return out;
}

inline bool TpQosProtocolCommitsMappedSl(TpProtocol proto)
{
    return proto == TpProtocol::TP || proto == TpProtocol::UBOE;
}

uint32_t TpQosCalSlAvailableCnt(uint32_t mask);
uint32_t TpQosSlValueAtRankInMask16(uint32_t mask, uint32_t rank);
uint16_t TpQosReadSlAvailableMask16(const struct TpAttr &attr);

/// 根据 hcclQos 与 slBitmap 选择 TP 列表下标及映射 SL；失败返回 false
bool TpQosApplySlPolicy(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag = "TpQos");

/// 校验 slMask、应用策略并检查 tpListIndex 范围
HcclResult TpQosSelectTpListEntry(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask,
    uint32_t &tpListIndexOut, uint32_t &mappedSlOut, const char *logTag = "TpQos");

uint8_t TpQosResolveUboeDscpLookupQos(const TpQosPolicyInput &policy, uint32_t nTp, uint16_t slMask);

bool TpQosGetDscpByQosFromHccnCfg(uint32_t devPhyId, uint8_t qos, uint8_t &dscpOut);

/// GetTpAttr 启动阶段 attrBitmap（含 sl_available / SL / UBOE dscp 位）
uint32_t TpQosBuildBootstrapAttrBitmap(TpProtocol tpProtocol);

/// Host 同步路径用 HOST_NET ctx；Device 异步路径用 GetByIp
RdmaHandle TpQosResolveUbRdmaHandle(bool isSync, uint32_t devPhyId, const IpAddress &locAddr);

/// Host 同步 GetTpAttr（RaCtxGetTpAttr）
HcclResult TpQosSyncGetTpAttr(RdmaHandle rdmaHandle, uint64_t tpHandle, TpProtocol tpProtocol,
    struct TpAttr &tpAttr, uint32_t &attrBitmap, const char *logTag = "TpQos");

HcclResult TpQosCommitMappedSlToTpAttr(RdmaHandle rdmaHandle, uint64_t tpHandle, uint32_t mappedSl,
    const char *logTag = "TpQos");

/// 将 SL 写回 TP；UBOE 另一次 SetTpAttr（kTpQosAttrBitmapUboeNetWithDscp，sip/dip/dscp 等网络属性一并下发）
/// isSync=true：Host 走 RaCtxSetTpAttr；false：Device 走 HrtRaSetTpAttrAsync（PCIe 标卡跳过；CTP 不写 SL）
HcclResult TpQosCommitAttrsAfterSlMapping(RdmaHandle rdmaHandle, bool isPcieStd, const TpQosPolicyInput &policy,
    const struct TpAttr &tpAttr, uint64_t tpHandle, uint32_t mappedSl, uint32_t nTp, uint16_t slMask,
    uint32_t devPhyId, const char *logTag = "TpQos", bool isSync = false);

} // namespace Hccl

#endif // HCCL_TP_QOS_H
