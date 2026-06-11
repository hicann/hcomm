/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CHANNEL_PROCESS_H
#define CHANNEL_PROCESS_H

#include "hcomm_c_adpt.h"
#include "channel.h"
#include "mem_host_pub.h"
#include <mutex>
#include <string>

namespace hcomm {

struct DeviceChannelKey {
    int32_t deviceId;
    ChannelHandle handle;

    bool operator==(const DeviceChannelKey& other) const {
        return deviceId == other.deviceId && handle == other.handle;
    }
};

struct DeviceChannelKeyHash {
    std::size_t operator()(const DeviceChannelKey& key) const {
        return std::hash<int32_t>()(key.deviceId) ^
               (std::hash<ChannelHandle>()(key.handle) << 1);
    }
};

class ChannelProcess {
public:
    ChannelProcess() = default;
    ~ChannelProcess() = default;
    static HcclResult CreateChannelsLoop(EndpointHandle endpointHandle, CommEngine engine,
        HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *outHandles);
    static HcclResult ChannelUpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum, ChannelHandle channelHandle);
    static HcclResult ConnectChannels(ChannelHandle* targetChannels, uint32_t channelNum, CommEngine engine);
    static HcclResult SaveChannels(ChannelHandle* targetChannels, ChannelHandle* userChannels, 
        HcommChannelDesc *channelDescs, uint32_t channelNum, CommEngine engine, aclrtBinHandle binHandle);
    static HcclResult ChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList);
    static HcclResult ChannelKernelLaunchForComm(ChannelHandle *channelHandles, ChannelHandle *hostChannelHandles,
        HcommChannelDesc* hcommDesc, uint32_t listNum, const std::string &commTag, aclrtBinHandle binHandle);
    static HcclResult ChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t *notifyNum);
    static HcclResult ChannelGetRemoteMem(ChannelHandle channelHandle, CommMem **remoteMem, uint32_t *memNum, char **memTags);
    static HcclResult ChannelGetUserRemoteMem(ChannelHandle channelHandle, CommMem **remoteMem, char ***memTag, uint32_t *memNum);
    static HcclResult ChannelKernelDestroy(ChannelHandle *channelHandles, uint32_t listNum, aclrtBinHandle binHandle);
    static HcclResult ChannelDestroy(const ChannelHandle *channels, uint32_t channelNum, aclrtBinHandle binHandle = nullptr);
    static HcclResult ChannelGet(const ChannelHandle channelHandle, void **channel);

    static HcclResult ChannelClean(const ChannelHandle *channelList, uint32_t listNum);
    static HcclResult ChannelResume(const ChannelHandle *channelList, uint32_t channelNum);
    static HcclResult ChannelUpdateKernelLaunch(ChannelHandle* deviceChannelHandles, ChannelHandle* hostChannelHandles, 
        uint32_t listNum, const std::string &commTag, aclrtBinHandle binHandle);
    
private:
    template <typename Func>
    static HcclResult WithChannelByHandleLocked(ChannelHandle inHandle, Func &&func);

    static HcclResult CombineHostMemory(const std::vector<std::vector<char>> &hostPackBuffers, 
        hccl::HostMem &hostPackBuf);
    static HcclResult FillChannelD2HMap(ChannelHandle *deviceChannelHandles, ChannelHandle *hostChannelHandles, 
        uint32_t listNum);
    static HcclResult LaunchChannelKernelCommon(ChannelHandle *channelHandles, ChannelHandle *hostChannelHandles,
        HcommChannelDesc* hcommDesc, uint32_t listNum, const std::string &commTag, aclrtBinHandle binHandle,
        const std::string &kernelName, bool needProfiling);
    static HcclResult ChannelKernelLaunchForBase(ChannelHandle *channelHandles, ChannelHandle *hostChannelHandles,
        HcommChannelDesc* hcommDesc, uint32_t listNum, aclrtBinHandle binHandle);
    static HcclResult LaunchChannelKernel(ChannelHandle *channelHandles, ChannelHandle *hostChannelHandles,
        HcommChannelDesc* hcommDesc, uint32_t listNum, aclrtBinHandle binHandle);
    static HcclResult LaunchCommonChannelKernel(ChannelHandle *channelHandles,
        ChannelHandle *hostChannelHandles, uint32_t listNum, HcommChannelKind channelKind, aclrtBinHandle binHandle);

    static HcclResult ChannelResumeConcurrency(const ChannelHandle *channelList, uint32_t channelNum);
    static HcclResult RemoveSingleChannel(int32_t deviceId, ChannelHandle inHandle,
        std::vector<ChannelHandle> &deviceHandles);

    static std::unordered_map<ChannelHandle, std::shared_ptr<Channel>> g_ChannelMap;
    static std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> g_ChannelD2HMap;
    static std::mutex g_ChannelMapMtx;
};
}
#endif // CHANNEL_PROCESS_H
