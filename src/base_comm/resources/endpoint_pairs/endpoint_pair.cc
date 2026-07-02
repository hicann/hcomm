/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_pair.h"
#include "socket_config.h"
#include "hcomm_c_adpt.h"
#include "orion_adpt_utils.h"
#include "channel_process.h"
#include "comm_engine_utils.h"


#include "hcom_common.h"
#include "exception_handler.h"

namespace hcomm {

EndpointPair::~EndpointPair() 
{
    for (auto &channels : channelHandles_) {
        if (channels.second.empty()) {
            continue;
        }
        (void)ChannelProcess::ChannelDestroy(channels.second.data(), channels.second.size());
    }
}

HcclResult EndpointPair::Init()
{
    EXCEPTION_CATCH(socketMgr_ = std::make_unique<SocketMgr>(), return HCCL_E_PTR);
    channelHandles_.clear();
    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));

    return HCCL_SUCCESS;
}

HcclResult EndpointPair::GetHostSocketWithRank(const uint32_t myRank, const uint32_t rmtRank, const std::string &socketTag,
    const uint32_t listenPort, u32 reuseIdx, Hccl::Socket*& socket)
{
    uint32_t connectMode = 0;
    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEndpointDesc_, remoteEndpointDesc_, linkData, reuseIdx));
    std::string linkTag = socketTag;
    if (linkData.GetReuseIdx() != "0") {
        linkTag += ("_" + linkData.GetReuseIdx());
    }

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_910B && localEndpointDesc_.loc.locType != remoteEndpointDesc_.loc.locType) {
        connectMode = 1;
    }

    /* A2: host nic(cpu roce channel) -- device nic(transport ibv)时，两边ip地址格式不一样，判断大小算法不匹配
     * 修改成按照rank id大小判断server和client */
    Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, listenPort, linkTag, connectMode, myRank, rmtRank);
    CHK_RET(socketMgr_->GetHostSocket(socketConfig, socket));
    return HCCL_SUCCESS;
}

HcclResult EndpointPair::EnsureSocketMgrCompat(const uint32_t myRank, const std::string &socketTag)
{
    if (!socketMgrCompat_) {
        int32_t devLogicId = HcclGetThreadDeviceId();
        uint32_t devPhyId{0};
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(devLogicId), devPhyId));
        EXCEPTION_CATCH(socketMgrCompat_ =
            std::make_unique<Hccl::SocketManager>(myRank, devPhyId, devLogicId, socketTag),
            return HCCL_E_PTR);
        CHK_PTR_NULL(rankIpPortMap_);
        socketMgrCompat_->SetDeviceServerListenPortMap(*rankIpPortMap_);
    }
    return HCCL_SUCCESS;
}

Hccl::SocketConfig EndpointPair::BuildSocketConfig(const Hccl::LinkData &linkData, const std::string &socketTag)
{
    std::string linkTag = socketTag;
    if (linkData.GetReuseIdx() != "0") {
        linkTag += ("_" + linkData.GetReuseIdx());
    }
    return Hccl::SocketConfig(linkData.GetRemoteRankId(), linkData, linkTag);
}

HcclResult EndpointPair::HandleHostSocketOrBuildLinkData(const uint32_t myRank, const uint32_t rmtRank,
    const std::string &socketTag, u32 reuseIdx, const uint32_t listenPort, Hccl::Socket*& socket,
    uint32_t devicePhyId, uint32_t remoteDevicePhyId, Hccl::LinkData &linkData, bool &isHost)
{
    if (localEndpointDesc_.loc.locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        std::string socketTagPrefix = socketTag;
        if (myRank <= rmtRank) {
            socketTagPrefix += "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank);
        } else {
            socketTagPrefix += "_" + std::to_string(rmtRank) + "_" + std::to_string(myRank);
        }
        CHK_RET(this->GetHostSocketWithRank(myRank, rmtRank, socketTagPrefix, listenPort, reuseIdx, socket));
        isHost = true;
        return HCCL_SUCCESS;
    }
    isHost = false;
    CHK_RET(EndpointDescPairToLinkDataWithRankIds(myRank, rmtRank,
        localEndpointDesc_, remoteEndpointDesc_, linkData, devicePhyId, remoteDevicePhyId, reuseIdx));
    return HCCL_SUCCESS;
}

HcclResult EndpointPair::GetSocketInternal(const uint32_t myRank, const uint32_t rmtRank,
    const std::string &socketTag, u32 reuseIdx, const uint32_t listenPort, Hccl::Socket*& socket,
    uint32_t devicePhyId, uint32_t remoteDevicePhyId, bool connectMode)
{
    Hccl::LinkData linkData = BuildDefaultLinkData();
    bool isHost = false;
    CHK_RET(HandleHostSocketOrBuildLinkData(myRank, rmtRank, socketTag, reuseIdx, listenPort, socket,
        devicePhyId, remoteDevicePhyId, linkData, isHost));
    if (isHost) {
        return HCCL_SUCCESS;
    }
    EXCEPTION_HANDLE_BEGIN
    Hccl::SocketConfig socketConfig = BuildSocketConfig(linkData, socketTag);
    if (connectMode) {
        CHK_PTR_NULL(socketMgrCompat_);
        socketMgrCompat_->ConnectSockets(socketConfig);
    } else {
        CHK_RET(EnsureSocketMgrCompat(myRank, socketTag));
        socketMgrCompat_->BatchCreateSockets(socketConfig);
    }
    socket = socketMgrCompat_->GetConnectedSocket(socketConfig);
    CHK_PTR_NULL(socket);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult EndpointPair::ServerInit(const uint32_t myRank, const uint32_t rmtRank,
    const std::string &socketTag, u32 reuseIdx, uint32_t devicePhyId, uint32_t remoteDevicePhyId)
{
    if (localEndpointDesc_.loc.locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        // host网卡不走device的socket监听
        return HCCL_SUCCESS;
    }
    // server监听
    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkDataWithRankIds(myRank, rmtRank, localEndpointDesc_,
        remoteEndpointDesc_, linkData, devicePhyId, remoteDevicePhyId, reuseIdx));
    EXCEPTION_HANDLE_BEGIN
    CHK_RET(EnsureSocketMgrCompat(myRank, socketTag));
    Hccl::SocketConfig socketConfig = BuildSocketConfig(linkData, socketTag);
    // 调用sock的server监听接口
    socketMgrCompat_->ServerListen(socketConfig);
    EXCEPTION_HANDLE_END

    return HCCL_SUCCESS;
}

HcclResult EndpointPair::GetConnectedSocket(const uint32_t myRank, const uint32_t rmtRank,
    const std::string &socketTag, u32 reuseIdx, const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId)
{
    // 该接口内进行建链和获取socket
    return GetSocketInternal(myRank, rmtRank, socketTag, reuseIdx, listenPort, socket,
        devicePhyId, remoteDevicePhyId, true);
}

HcclResult EndpointPair::GetSocket(const uint32_t myRank, const uint32_t rmtRank,
    const std::string &socketTag, u32 reuseIdx, const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId)
{
    // 临时方案：支持混跑新增，非Roce场景走orion socketMgr实现server socket复用
    return GetSocketInternal(myRank, rmtRank, socketTag, reuseIdx, listenPort, socket,
        devicePhyId, remoteDevicePhyId, false);
}

HcclResult EndpointPair::CreateChannel(EndpointHandle endpointHandle, CommEngine engine, u32 reuseIdx,
        HcommChannelDesc *channelDescs, ChannelHandle *channels)
{
    if (channelHandles_.find(engine) == channelHandles_.end() || channelHandles_[engine].size() <= reuseIdx) {
        CHK_RET_UNAVAIL(static_cast<HcclResult>(
            HcommCollectiveChannelCreate(endpointHandle, engine, channelDescs, 1, channels)));
        channelHandles_[engine].push_back(channels[0]);
        return HCCL_SUCCESS;
    }

    channels[0] = channelHandles_[engine][reuseIdx];
    if (channelDescs->memHandleNum > 1) {
        CHK_RET(static_cast<HcclResult>(HcommChannelUpdateMemInfo(channelDescs->memHandles + 1, channelDescs->memHandleNum - 1, channels[0])));
    }
    return HCCL_SUCCESS;
}

// 找到对应的channelhandle，调用HcommChannelDestroy销毁平台层对象，并删除channelHandles_中的channelHandle元素
HcclResult EndpointPair::DestroyChannel(CommEngine engine, u32 reuseIdx)
{
    if (IsChannelNotExist(engine, reuseIdx)) {
        HCCL_WARNING("EndpointPair::DestroyChannel: engine[%s] reuseIdx[%u], channelHandle size[%u],"
                     "channel not found, skip destroy channel", GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx, channelHandles_[engine].size());
        return HCCL_SUCCESS;
    }
    HCCL_INFO("EndpointPair::DestroyChannel: engine[%s] reuseIdx[%u], channelHandle size[%u],"
              "start destroy channel", GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx, channelHandles_[engine].size());
    ChannelHandle channelHandle = channelHandles_[engine][reuseIdx];
    CHK_RET(static_cast<HcclResult>(HcommChannelDestroy(&channelHandle, 1)));
    // 去掉channelHandles_中reuseIdx位置的channelHandle
    channelHandles_[engine].erase(channelHandles_[engine].begin() + reuseIdx);
    HCCL_INFO("EndpointPair::DestroyChannel: engine[%s] reuseIdx[%u] destroy channel success,"
              "channelHandle size[%u]", GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx, channelHandles_[engine].size());
    return HCCL_SUCCESS;
}

// 检查channel是否存在，channel不存在则返回true
bool EndpointPair::IsChannelNotExist(CommEngine engine, u32 reuseIdx)
{
    return channelHandles_.find(engine) == channelHandles_.end() || channelHandles_[engine].size() <= reuseIdx;
}

const std::unordered_map<CommEngine, std::vector<ChannelHandle>>& EndpointPair::GetChannelHandles()
{
    return channelHandles_;
}

} // namespace hcomm
