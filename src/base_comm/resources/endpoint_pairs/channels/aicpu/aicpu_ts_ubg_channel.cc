/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_ubg_channel.h"
#include "orion_adpt_utils.h"
#include "endpoint.h"

// Orion
#include "topo_common_types.h"
#include "tp_manager.h"

namespace hcomm {

HcclResult AicpuTsUbgChannel::Init()
{
    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    CHK_RET(ParseInputParam());

    // UBG 直接从 EID type CommAddr 获取地址，不做 IP→EID 转换
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, locAddr_));
    CHK_RET(CommAddrToIpAddress(remoteEp_.commAddr, rmtAddr_));
    HCCL_INFO("[AicpuTsUbgChannel][%s] locAddr_[%s], rmtAddr_[%s]",
        __func__, locAddr_.Describe().c_str(), rmtAddr_.Describe().c_str());

    CHK_RET(BuildSocket());
    CHK_RET(BuildNotify());
    /*
        HccpRaGetDevBaseAttr
        获取urma read/write 单个wr的最大传输数据大小
        调用前,rdmaHandle_要在ParseInputParam中被赋值好,之后BuildConnection会使用获取的属性
        ubg的BuildConnection不再Init里面执行，Init之后会有单独流程建链
    */
    CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUbgChannel::BuildConnection()
{
    Hccl::OpMode        opMode = Hccl::OpMode::OPBASE;
    bool                devUsed  = true; // aicpu 为 true
    Hccl::LinkProtocol  protocol;
    CHK_RET(CommProtocolToLinkProtocol(localEp_.protocol, protocol));

    // UBG 的 locAddr_/rmtAddr_ 已经是 EID-based IpAddress，无需额外转换
    HCCL_INFO("[AicpuTsUbgChannel][%s] LinkProtocol[%s], locAddr_[%s], rmtAddr_[%s]",
        __func__, protocol.Describe().c_str(), locAddr_.Describe().c_str(), rmtAddr_.Describe().c_str());

    s32 deviceLogicId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    Hccl::TpManager::GetInstance(deviceLogicId).Init();

    // UBG 使用 DevUbUbgConnection，locAddr_/rmtAddr_ 作为 EID 地址
    std::unique_ptr<Hccl::DevUbConnection> ubConn = std::make_unique<Hccl::DevUbUbgConnection>(rdmaHandle_,
        locAddr_, rmtAddr_, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL, locAddr_, rmtAddr_);
    CHK_SMART_PTR_NULL(ubConn);

    if (devBaseAttr_.maxReadSize == 0 || devBaseAttr_.maxWriteSize == 0) {
        HCCL_ERROR("[AicpuTsUbgChannel][%s] maxReadSize[%u] or maxWriteSize[%u] must not be zero", __func__,
            devBaseAttr_.maxReadSize, devBaseAttr_.maxWriteSize);
        return HCCL_E_PARA;
    }
    ubConn->SetMaxReadSize(devBaseAttr_.maxReadSize);
    ubConn->SetMaxWriteSize(devBaseAttr_.maxWriteSize);
    HCCL_INFO("[AicpuTsUbgChannel][%s] maxReadSize[%u], maxWriteSize[%u]", __func__, devBaseAttr_.maxReadSize,
        devBaseAttr_.maxWriteSize);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));
    return HCCL_SUCCESS;
}

void AicpuTsUbgChannel::SendFinish()
{
    HCCL_INFO("start send Finish Msg [%s]", UBG_FINISH_MSG);
    sendFinishMsg_ = std::vector<char>(UBG_FINISH_MSG, UBG_FINISH_MSG + FINISH_MSG_SIZE);
    socket_->SendAsync(sendFinishMsg_.data(), FINISH_MSG_SIZE);
    HCCL_INFO("end send Finish Msg [%s]", UBG_FINISH_MSG);
}

void AicpuTsUbgChannel::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    HCCL_INFO("start recv Finish Msg [%s]", UBG_FINISH_MSG);
    socket_->RecvAsync(reinterpret_cast<u8 *>(recvFinishMsg_.data()), FINISH_MSG_SIZE);
    HCCL_INFO("end recv Finish Msg [%s]", UBG_FINISH_MSG);
}

void AicpuTsUbgChannel::ProcessUbgState()
{
    auto SetState = [&](UbgStatus next, ChannelStatus ch) { ubgStatus = next; channelStatus = ch; };

    switch (ubgStatus) {
        case UbgStatus::INIT:
            SetState(UbgStatus::BUILD_CONN, channelStatus);
            break;
        case UbgStatus::BUILD_CONN:
            BuildConn(); SetState(UbgStatus::SEND_SIZE, channelStatus);
            break;
        case UbgStatus::SEND_SIZE:
            if (IsResReady()) { SendDataSize(); SetState(UbgStatus::RECV_SIZE, channelStatus); }
            break;
        case UbgStatus::RECV_SIZE:
            RecvDataSize(); SetState(isRecvFirst_ ? UbgStatus::RECV_DATA : UbgStatus::SEND_DATA, channelStatus);
            break;
        case UbgStatus::SEND_DATA:
            SendExchangeData(); SetState(isRecvFirst_ ? UbgStatus::PROCESS_DATA : UbgStatus::RECV_DATA, channelStatus);
            break;
        case UbgStatus::RECV_DATA:
            RecvExchangeData(); SetState(isRecvFirst_ ? UbgStatus::SEND_DATA : UbgStatus::PROCESS_DATA, channelStatus);
            break;
        case UbgStatus::PROCESS_DATA:
            if (RecvDataProcess()) {
                ubgStatus = UbgStatus::SEND_FIN;
            } else {
                channelStatus = ChannelStatus::READY;
                ubgStatus = UbgStatus::READY;
            }
            break;
        case UbgStatus::SEND_FIN:
            if (IsConnsReady()) { SendFinish(); SetState(UbgStatus::RECV_FIN, channelStatus); }
            break;
        case UbgStatus::RECV_FIN:
            RecvFinish(); SetState(UbgStatus::SET_READY, channelStatus);
            break;
        case UbgStatus::SET_READY:
            channelStatus = ChannelStatus::READY; SetState(UbgStatus::READY, ChannelStatus::READY);
            break;
        default:
            break;
    }
}

ChannelStatus AicpuTsUbgChannel::GetStatus()
{
    if (channelStatus == ChannelStatus::READY) {
        return channelStatus;
    }
    if (channelStatus == ChannelStatus::INIT) ubgStatus = UbgStatus::INIT;

    if (!IsSocketReady()) return channelStatus;

    ProcessUbgState();
    if (channelStatus == ChannelStatus::READY && channelDesc_.socket == nullptr && socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }

    return channelStatus;
}

} // namespace hcomm
