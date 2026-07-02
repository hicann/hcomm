/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <unordered_map>

#include "log.h"
#include "channel.h"
#include "./aicpu/aicpu_ts_urma_channel.h"
#include "comm_engine_utils.h"

#include "./aicpu/aicpu_ts_p2p_channel.h"
#include "./aicpu/aicpu_ts_uboe_channel.h"
#include "./aicpu/aicpu_ts_roce_channel.h"
#include "./host/host_cpu_roce_channel.h"
#include "./host/host_cpu_urma_channel.h"
#include "./ccu/ccu_urma_channel.h"
#include "./aiv/aiv_ub_mem_channel.h"
#include "./aiv/aiv_urma_channel.h"
#include "./aicpu/aicpu_ts_hccs_channel.h"
#include "./aicpu/aicpu_ts_roce_channel_v2.h"

namespace hcomm {
std::unordered_map<ChannelHandle, ChannelHandle> channelD2HHandleMap_;

HcclResult Channel::CreateChannel(
    EndpointHandle endpointHandle, CommEngine engine, 
    HcommChannelDesc channelDesc, std::shared_ptr<Channel>& channelPtr)
{
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    std::shared_ptr<Channel> uniqueChannelPtr;
    switch (engine) {
        case COMM_ENGINE_CPU:
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE) {
                EXCEPTION_CATCH(uniqueChannelPtr = std::make_unique<HostCpuRoceChannel>(endpointHandle, channelDesc),
                    return HCCL_E_PARA);
                break;
            }
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_CTP ||
                channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP) {
                EXCEPTION_CATCH(uniqueChannelPtr = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc),
                    return HCCL_E_PARA);
                break;
            }
            HCCL_ERROR("[Channel][%s] Engine[COMM_ENGINE_CPU] not support Protocol[%d]",
                        __func__, channelDesc.remoteEndpoint.protocol);
            return HCCL_E_NOT_SUPPORT;
        case COMM_ENGINE_CPU_TS:
            HCCL_ERROR("[Channel][%s] CommEngine[COMM_ENGINE_CPU_TS] not support", __func__);
            return HCCL_E_NOT_SUPPORT;
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBOE) {
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsUboeChannel(endpointHandle, channelDesc));
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_PCIE) {
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsP2pChannel(endpointHandle, channelDesc));
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE) {
                if (deviceType == DevType::DEV_TYPE_950) {
                    uniqueChannelPtr = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, engine);
                } else {
                    uniqueChannelPtr = std::make_unique<AicpuTsRoceChannel>(endpointHandle, channelDesc);
                }
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_CTP ||
                       channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP) {
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsUrmaChannel(endpointHandle, channelDesc));
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_HCCS) {
                uniqueChannelPtr.reset(
                    new (std::nothrow) AicpuTsHccsChannel(endpointHandle, channelDesc));
            } else {
                HCCL_ERROR("[Channel][%s] invalid protocol for engine %s, protocol=%d",
                    __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelDesc.remoteEndpoint.protocol);
                return HCCL_E_NOT_SUPPORT;
            }
            break;
        case COMM_ENGINE_AIV:
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE && deviceType == DevType::DEV_TYPE_950) {
                uniqueChannelPtr = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, engine);
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_CTP ||
                       channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP) {
                uniqueChannelPtr.reset(new (std::nothrow) AivUrmaChannel(endpointHandle, channelDesc));
            } else {
                uniqueChannelPtr.reset(new (std::nothrow) AivUbMemChannel(endpointHandle, channelDesc));
            }
            break;
        case COMM_ENGINE_CCU:
            uniqueChannelPtr.reset(
                new (std::nothrow) CcuUrmaChannel(endpointHandle, channelDesc));
            break;
        default:
            HCCL_ERROR("[Channel][%s] invalid type of CommEngine", __func__);
            return HCCL_E_NOT_FOUND;
    }
    CHK_PTR_NULL(uniqueChannelPtr);
    CHK_RET_UNAVAIL(uniqueChannelPtr->Init());
    channelPtr = std::move(uniqueChannelPtr);
    return HCCL_SUCCESS;
}

ChannelStatus Channel::TransportStatusToChannelStatus(Hccl::TransportStatus ts)
{
    switch (ts) {
        case Hccl::TransportStatus::INIT:
            return ChannelStatus::INIT;
        case Hccl::TransportStatus::SOCKET_OK:
            return ChannelStatus::SOCKET_OK;
        case Hccl::TransportStatus::SOCKET_TIMEOUT:
            return ChannelStatus::SOCKET_TIMEOUT;
        case Hccl::TransportStatus::READY:
            return ChannelStatus::READY;
        default:
            HCCL_ERROR("[Channel][%s] Invalid TransportStatus[%d]", __func__, ts);
            return ChannelStatus::FAILED;
    }
}

HcclResult Channel::UpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum)
{
    HCCL_WARNING("[UpdateMemInfo] not support.");
    return HCCL_SUCCESS;
}

HcommChannelKind Channel::GetChannelKind() const
{
    return channelKind_;
}

HcclResult Channel::Serialize(std::shared_ptr<hccl::DeviceMem> &out)
{
    out.reset();
    return HCCL_E_NOT_SUPPORT;
}

void Channel::AddPtrArrayDevMem(std::shared_ptr<hccl::DeviceMem> ptrArrayMem)
{
    if (ptrArrayMem == nullptr || !(*ptrArrayMem)) {
        HCCL_WARNING("[Channel][%s] invalid ptrArrayMem.", __func__);
        return;
    }
    ptrArrayDevMems_.push_back(std::move(ptrArrayMem));
}

void Channel::ReleasePtrArrayDevMems()
{
    ptrArrayDevMems_.clear();
}
} // namespace hcomm
