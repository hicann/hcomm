/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_comm_pub.h"
#include "hccl_communicator.h"

namespace hccl {
HcclResult hcclComm::RegistTaskAbortHandler() const { return HCCL_SUCCESS; }

HcclResult hcclComm::UnRegistTaskAbortHandler() const { return HCCL_SUCCESS; }

HcclResult hcclComm::GetOneSidedService(IHcclOneSidedService** service) { return HCCL_SUCCESS; }
HcclResult hcclComm::InitOneSidedServiceNetDevCtx(u32 remoteRankId) { return HCCL_SUCCESS; }
HcclResult hcclComm::OneSidedServiceStartListen(NicType nicType, HcclNetDevCtx netDevCtx) { return HCCL_SUCCESS; }
HcclResult hcclComm::GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress& ipAddress, u32& port)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::DeinitOneSidedService() { return HCCL_SUCCESS; }

HcclResult hcclComm::RegisterCommUserMem(void* addr, u64 size, void** handle) { return HCCL_SUCCESS; }

HcclResult hcclComm::DeregisterCommUserMem(void* handle) { return HCCL_SUCCESS; }

HcclResult hcclComm::ExchangeCommUserMem(void* handle, std::vector<u32>& peerRanks) { return HCCL_SUCCESS; }

HcclResult hcclComm::SetIndependentOpConfig(const CommConfig& commConfig, const RankTable_t& rankTable)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitIndependentOp() { return HCCL_SUCCESS; }

HcclResult hcclComm::PrepareChannelMem(const std::string& tag, TransportIOMem& transMem) { return HCCL_SUCCESS; }
HcclResult hcclComm::IndOpTransportAlloc(const std::string& tag, OpCommTransport& opCommTransport, bool isAicpuModeEn)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::CommGetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum) { return HCCL_SUCCESS; }

HcclResult hcclComm::CommGetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) { return HCCL_SUCCESS; }

HcclResult hcclComm::CommGetInstTopoTypeByNetLayer(uint32_t netLayer, u32* topoType) { return HCCL_SUCCESS; }
HcclResult hcclComm::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum) { return HCCL_SUCCESS; }

HcclResult hcclComm::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) { return HCCL_SUCCESS; }

HcclResult hcclComm::GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType) { return HCCL_SUCCESS; }

HcclResult hcclComm::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetRankGraph(GraphType type, void** graph, uint32_t* len) { return HCCL_SUCCESS; }

HcclResult
hcclComm::GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetHeterogMode(HcclHeterogMode* mode) { return HCCL_SUCCESS; }

HcclComm hcclComm::GetCommunicatorV2()
{
    HCCL_ERROR("[HcclComm][GetCommunicatorV2]collComm_ is nullptr");
    return nullptr;
}

void hcclComm::BinaryUnLoad() { binHandle_ = nullptr; }

HcclResult hcclComm::Resume()
{
    if (!IsCommunicatorV2()) {
        CHK_RET(communicator_->Resume());
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
