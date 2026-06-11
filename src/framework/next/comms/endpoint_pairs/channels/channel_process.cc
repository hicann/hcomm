/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel_process.h"
#include <cstdint>
#include <memory>
#include <vector>
#include "exception_handler.h"
#include "channel_param.h"
#include "aicpu_ts_urma_channel.h"
#include "aicpu_ts_uboe_channel.h"
#include "launch_aicpu.h"
#include "hcclCommDfx.h"
#include "env_config/env_config.h"
#include "aicpu_ts_p2p_channel.h"
#include "mem_device_pub.h"

namespace hcomm {

std::unordered_map<ChannelHandle, std::shared_ptr<Channel>> ChannelProcess::g_ChannelMap;
std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> ChannelProcess::g_ChannelD2HMap;
std::mutex ChannelProcess::g_ChannelMapMtx;

template <typename Func>
HcclResult ChannelProcess::WithChannelByHandleLocked(ChannelHandle inHandle, Func &&func)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::shared_ptr<Channel> channelPtr = nullptr;
    {
        // 单锁：该锁同时保护 g_ChannelD2HMap 和 g_ChannelMap
        std::lock_guard<std::mutex> lock(g_ChannelMapMtx);

        // 1) D2H 映射
        DeviceChannelKey key{deviceId, inHandle};
        auto itH = g_ChannelD2HMap.find(key);
        if (itH == g_ChannelD2HMap.end()) {
            HCCL_ERROR("[%s] handle not found in g_ChannelD2HMap, deviceId[%d], inHandle[0x%llx].", __func__, deviceId, inHandle);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        const ChannelHandle mappedHandle = itH->second;

        // 2) ChannelMap 查找
        auto itC = g_ChannelMap.find(mappedHandle);
        if (itC == g_ChannelMap.end() || !itC->second) {
            HCCL_ERROR("[%s] channel not found in g_ChannelMap, deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx].",
                __func__,
                deviceId,
                inHandle,
                mappedHandle);
            return HcclResult::HCCL_E_INTERNAL;
        }

        channelPtr = itC->second;
        if (channelPtr == nullptr) {
            HCCL_ERROR(
                "[%s] null channel pointer, deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx].", __func__, deviceId, inHandle, mappedHandle);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    // 3) 执行用户逻辑
    return std::forward<Func>(func)(*channelPtr);
}

HcclResult ChannelProcess::CreateChannelsLoop(EndpointHandle endpointHandle, CommEngine engine,
    HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *outHandles)
{
    CHK_PTR_NULL(endpointHandle);

    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    for (uint32_t i = 0; i < channelNum; ++i) {
        std::shared_ptr<Channel> tmpPtr = nullptr;
        CHK_RET_UNAVAIL(Channel::CreateChannel(endpointHandle, engine, channelDescs[i], tmpPtr));
        CHK_SMART_PTR_NULL(tmpPtr);

        ChannelHandle handle = reinterpret_cast<ChannelHandle>(tmpPtr.get());
        outHandles[i] = handle;
        HCCL_INFO("%s deviceId[%d], handle[0x%llx], ptr[%p]", __func__, deviceId, handle, tmpPtr.get());

        // 仅在修改全局表时持锁
        {
            std::lock_guard<std::mutex> lock(g_ChannelMapMtx);

            if (g_ChannelMap.find(handle) != g_ChannelMap.end()) {
                HCCL_ERROR("[%s] channel handle already exists [0x%llx] in ChannelMap", __func__, handle);
                return HCCL_E_INTERNAL;
            }
            DeviceChannelKey key{deviceId, handle};
            if (g_ChannelD2HMap.find(key) != g_ChannelD2HMap.end()) {
                HCCL_ERROR("[%s] channel handle already exists deviceId[%d], handle[0x%llx] in g_ChannelD2HMap", __func__, deviceId, handle);
                return HCCL_E_INTERNAL;
            }

            g_ChannelMap.emplace(handle, std::move(tmpPtr));
            g_ChannelD2HMap.emplace(key, handle);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelUpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum, ChannelHandle channelHandle)
{
    EXCEPTION_HANDLE_BEGIN
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    Channel *channel = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
        // 1) D2H 映射
        DeviceChannelKey key{deviceId, channelHandle};
        auto itH = g_ChannelD2HMap.find(key);
        if (itH == g_ChannelD2HMap.end()) {
            HCCL_ERROR("[%s] handle not found in g_ChannelD2HMap, deviceId[%d], channelHandle[0x%llx].", __func__, deviceId, channelHandle);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        const ChannelHandle mappedHandle = itH->second;

        // 2) ChannelMap 查找
        auto itC = g_ChannelMap.find(mappedHandle);
        if (itC == g_ChannelMap.end() || !itC->second) {
            HCCL_ERROR("[%s] channel not found in g_ChannelMap, deviceId[%d], channelHandle[0x%llx], mappedHandle[0x%llx].",
                __func__,
                deviceId,
                channelHandle,
                mappedHandle);
            return HcclResult::HCCL_E_INTERNAL;
        }
        channel = itC->second.get();
    }
    // UpdateMemInfo需要rank间交互，若在锁内执行会导致单进程多线程场景其他rank被锁拦住
    CHK_RET(channel->UpdateMemInfo(memHandles, memHandleNum));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList)
{
    EXCEPTION_HANDLE_BEGIN

    // 不得随意添加无效日志，可能造成刷屏
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(statusList);

    u32 readyCount = 0;

    for (uint32_t i = 0; i < listNum; ++i) {
        const ChannelHandle inHandle = channelList[i];
        int32_t status = 0;

        // 单锁：D2H 映射 + 查 map + 锁内调用 GetStatus()
        HcclResult ret = WithChannelByHandleLocked(inHandle, [&](Channel &channel) -> HcclResult {
            status = channel.GetStatus();  // 锁内调用，防止 destroy 并发释放
            return HcclResult::HCCL_SUCCESS;
        });
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[%s] Get ChannelHandle failed.", __func__);
            return ret;
        }
        CHK_PRT_RET(
            status == ChannelStatus::FAILED, HCCL_ERROR("[%s] FAILED, status[%d]", __func__, status), HCCL_E_NETWORK);

        CHK_PRT_RET(status == ChannelStatus::SOCKET_TIMEOUT,
            HCCL_ERROR("[%s] TIMEOUT, status[%d]", __func__, status),
            HCCL_E_TIMEOUT);

        readyCount += (status == ChannelStatus::READY) ? 1 : 0;
        statusList[i] = status;
    }
    if (readyCount != listNum) {
        return HCCL_E_AGAIN;
    }
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ConnectChannels(ChannelHandle* targetChannels, uint32_t channelNum,
    CommEngine engine)
{
    CHK_PTR_NULL(targetChannels);
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);

    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();

    std::vector<int32_t> statusVec(channelNum, 0);
    int32_t* statusList = statusVec.data();

    while (true) {
        HcclResult ret = ChannelGetStatus(targetChannels, channelNum, statusList);
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[%s] channel connect timeout", __func__);
            return HCCL_E_TIMEOUT;
        }
        if (ret == HCCL_E_AGAIN) {
            continue;
        }
        if (ret == HCCL_SUCCESS) {
            break;
        } else {
            HCCL_ERROR("[%s] FAIL, return HcclResult[%d]", __func__, ret);
            return ret;
        }
    }

    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::CombineHostMemory(const std::vector<std::vector<char>> &hostPackBuffers, 
    hccl::HostMem &hostPackBuf)
{
    if (hostPackBuffers.empty()) {
        HCCL_ERROR("[%s] hostPackBuffers is empty, please check.", __func__);
        return HCCL_E_PARA;
    }

    // 将离散数据复制到连续内存中
    u8 *dstPtr = static_cast<u8 *>(hostPackBuf.ptr());  // 目标内存起始地址
    u64 dstMax = hostPackBuf.size();
    u64 packSize = 0;

    for (const auto &mem : hostPackBuffers) {
        packSize += mem.size();
        CHK_PRT_RET(packSize > dstMax,
            HCCL_ERROR("[%s] fail, packSize[%llu] is bigger than dstMax[%llu]", __func__, packSize, dstMax),
            HCCL_E_PARA);

        CHK_SAFETY_FUNC_RET(memcpy_s(dstPtr, mem.size(), mem.data(), mem.size()));
        dstPtr += mem.size();  // 移动目标指针
    }

    HCCL_INFO("[%s] end of merging host memory, hostPackBuf.addr[%p], hostPackBuf.size[%zu]",
        __func__,
        hostPackBuf.ptr(),
        hostPackBuf.size());

    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::FillChannelD2HMap(ChannelHandle *deviceChannelHandles,
    ChannelHandle *hostChannelHandles, uint32_t listNum)
{
    CHK_PTR_NULL(deviceChannelHandles);
    CHK_PTR_NULL(hostChannelHandles);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s]Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
    for (uint32_t idx = 0; idx < listNum; idx++) {
        auto deviceChannelHandle = deviceChannelHandles[idx];
        auto hostChannelHandle = hostChannelHandles[idx];
        HCCL_INFO("%s deviceId[%d], deviceChannelHandle[0x%llx], hostChannelHandle[0x%llx]",
            __func__,
            deviceId,
            deviceChannelHandle,
            hostChannelHandle);
        DeviceChannelKey key{deviceId, deviceChannelHandle};
        g_ChannelD2HMap.emplace(key, hostChannelHandle);
    }

    return HCCL_SUCCESS;
}

static HcclResult FillChannelParam(HcclChannelUrmaRes &channelParam, 
    const std::string &commTag, 
    hccl::DeviceMem &deviceChannelList,
    hccl::DeviceMem &devicePackBuf,
    uint32_t listNum, 
    uint32_t totalListNum,
    hccl::DeviceMem &channelSizeAddr)
{
    // channelParam资源参数填充
    s32 sRet = strncpy_s(channelParam.hcomId, HCOMID_MAX_LENGTH, commTag.c_str(), HCOMID_MAX_LENGTH - 1);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] str copy fail. return[%d]", __func__, sRet), HCCL_E_INTERNAL);

    channelParam.channelList = static_cast<void *>(deviceChannelList.ptr());
    channelParam.listNum = listNum;
    channelParam.uniqueIdAddr = static_cast<void *>(devicePackBuf.ptr());
    channelParam.uniqueIdSize = totalListNum;
    channelParam.channelSizeAddr = static_cast<void *>(channelSizeAddr.ptr());

    CHK_RET(hrtGetDevice(&channelParam.deviceLogicId));
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    channelParam.deviceType = static_cast<u32>(devType);

    return HCCL_SUCCESS;
}

template<typename T>
static HcclResult LaunchKernelDeviceParam(const T &channelParam, aclrtBinHandle binHandle, const std::string &kernelName)
{
    hccl::Stream localStream = hccl::Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    hccl::DeviceMem addr = hccl::DeviceMem::alloc(sizeof(T));
    CHK_PTR_NULL(addr.ptr());

    CHK_RET(hrtMemSyncCopy(addr.ptr(),
        sizeof(T),
        &channelParam,
        sizeof(T),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    uint64_t context = reinterpret_cast<uint64_t>(addr.ptr());

    CHK_RET(hccl::AicpuAclKernelLaunch(localStream.ptr(),
        reinterpret_cast<void *>(&context),
        sizeof(context),
        binHandle,
        kernelName,
        true,
        NOTIFY_DEFAULT_WAIT_TIME));

    CHK_RET(hcclStreamSynchronize(localStream.ptr(), 60));

    HCCL_INFO("[%s] kernel[%s] launch success.", __func__, kernelName.c_str());
    return HCCL_SUCCESS;
}

static HcclResult LaunchKernel(const HcclChannelUrmaRes &channelParam, aclrtBinHandle binHandle, const std::string &kernelName)
{
    return LaunchKernelDeviceParam(channelParam, binHandle, kernelName);
}

HcclResult ChannelProcess::LaunchChannelKernelCommon(ChannelHandle *channelHandles, ChannelHandle *hostChannelHandles,
    HcommChannelDesc* hcommDesc, uint32_t listNum, const std::string &commTag, aclrtBinHandle binHandle,
    const std::string &kernelName, bool needProfiling)
{
    CHK_PTR_NULL(channelHandles);
    CHK_PTR_NULL(hostChannelHandles);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s]Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    HCCL_RUN_INFO("[%s] listNum[%u], commTag[%s]", __func__, listNum, commTag.c_str());
    std::vector<std::vector<char>> hostPackBuffers(listNum);
    HcclChannelUrmaRes channelParam{};
    CHK_SAFETY_FUNC_RET(memset_s(&channelParam, sizeof(channelParam), 0, sizeof(channelParam)));

    // 获取host侧序列化的地址
    std::vector<u32> channelSizeVec{};
    uint32_t totalListNum = 0;
    for (uint32_t index = 0; index < listNum; index++) {
        if (hcommDesc[index].remoteEndpoint.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
            auto aicpuTsP2pChannel = reinterpret_cast<AicpuTsP2pChannel *>(hostChannelHandles[index]);
            CHK_PRT(aicpuTsP2pChannel->H2DResPack(hostPackBuffers[index]));
        } else if (hcommDesc[index].remoteEndpoint.protocol == CommProtocol::COMM_PROTOCOL_UBOE) {
            auto aicpuTsUboeChannel = reinterpret_cast<AicpuTsUboeChannel *>(hostChannelHandles[index]);
            CHK_PRT(aicpuTsUboeChannel->H2DResPack(hostPackBuffers[index]));
        } else {
            auto aicpuTsUrmaChannel = reinterpret_cast<AicpuTsUrmaChannel *>(hostChannelHandles[index]);
            CHK_PRT(aicpuTsUrmaChannel->H2DResPack(hostPackBuffers[index]));
        }
        totalListNum += hostPackBuffers[index].size();
        channelSizeVec.push_back(hostPackBuffers[index].size());
    }
    HCCL_INFO("[%s] totalListNum[%llu]", __func__, totalListNum);

    // 分配连续的host内存，将序列化的地址放入其中
    hccl::HostMem hostPackBuf = hccl::HostMem::alloc(totalListNum);
    CHK_PTR_NULL(hostPackBuf.ptr());
    CHK_RET(CombineHostMemory(hostPackBuffers, hostPackBuf));
    hccl::DeviceMem devicePackBuf = hccl::DeviceMem::alloc(totalListNum);
    CHK_PTR_NULL(devicePackBuf.ptr());

    // 将host侧序列化内容拷贝到device侧内存中
    CHK_RET(hrtMemSyncCopy(devicePackBuf.ptr(),
        totalListNum,
        hostPackBuf.ptr(),
        totalListNum,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    hccl::DeviceMem channelSizeAddr = hccl::DeviceMem::alloc(channelSizeVec.size() * sizeof(u32));
    CHK_PTR_NULL(channelSizeAddr.ptr());

    CHK_RET(hrtMemSyncCopy(channelSizeAddr.ptr(),
        channelSizeVec.size() * sizeof(u32),
        channelSizeVec.data(),
        channelSizeVec.size() * sizeof(u32),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    // 为device侧的channelList分配内存
    hccl::DeviceMem deviceChannelList = hccl::DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());

    // 填充channelParam参数
    CHK_RET(FillChannelParam(channelParam, commTag, deviceChannelList, devicePackBuf, 
        listNum, totalListNum, channelSizeAddr));
    
    // profiling信息
    hccl::DeviceMem remoteRankList = hccl::DeviceMem::alloc(listNum * sizeof(u32));
    CHK_PTR_NULL(remoteRankList.ptr());
    std::vector<u32> remoteRankIdList(listNum);
    // 集合通信场景才能开启
    if (needProfiling) {
        for ( u32 i = 0; i < listNum; ++i) {
            CHK_RET(hccl::HcclCommDfx::GetChannelRemoteRankId(commTag, hostChannelHandles[i], remoteRankIdList[i]));
        }
        // 通过安全的内存拷贝将主机内存数据传输到设备内存
        CHK_RET(hrtMemSyncCopy(remoteRankList.ptr(), listNum * sizeof(u32), remoteRankIdList.data(), 
                listNum * sizeof(u32), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        channelParam.remoteRankList = static_cast<u32 *>(remoteRankList.ptr());
    }

    // 调用抽离的通用内核启动函数
    CHK_RET(LaunchKernel(channelParam, binHandle, kernelName));

    // 将device侧的channelList拷贝回host侧的channelList
    CHK_RET(hrtMemSyncCopy(channelHandles,
        listNum * sizeof(ChannelHandle),
        deviceChannelList.ptr(),
        listNum * sizeof(ChannelHandle),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    CHK_RET(FillChannelD2HMap(channelHandles, hostChannelHandles, listNum));

    HCCL_INFO("[%s] channel kernel launch success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelKernelLaunchForComm(ChannelHandle *channelHandles, ChannelHandle *hostChannelHandles,
    HcommChannelDesc* hcommDesc, uint32_t listNum, const std::string &commTag, aclrtBinHandle binHandle)
{
    return LaunchChannelKernelCommon(channelHandles, hostChannelHandles, hcommDesc, listNum,
        commTag, binHandle, "RunAicpuIndOpChannelInitV2", true);
}

HcclResult ChannelProcess::ChannelKernelLaunchForBase(ChannelHandle *channelHandles, ChannelHandle *hostChannelHandles,
    HcommChannelDesc* hcommDesc, uint32_t listNum, aclrtBinHandle binHandle)
{
    return LaunchChannelKernelCommon(channelHandles, hostChannelHandles, hcommDesc, listNum, "", 
        binHandle, "RunAicpuChannelInitV2", false);
}

namespace {

struct KHost {
    std::vector<std::shared_ptr<hccl::DeviceMem>> mem;
    std::vector<void *> ptr;
    std::vector<u64> sz;
    std::vector<u32> kind;
};

struct KDev {
    hccl::DeviceMem data;
    hccl::DeviceMem size;
    hccl::DeviceMem type;
};

HcclResult PackHost(ChannelHandle *host, uint32_t n, HcommChannelKind k, KHost &out)
{
    out.mem.resize(n);
    out.ptr.resize(n);
    out.sz.resize(n);
    out.kind.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto *channel = reinterpret_cast<Channel *>(host[i]);
        CHK_PTR_NULL(channel);
        CHK_RET(channel->Serialize(out.mem[i]));
        CHK_PTR_NULL(out.mem[i]);
        CHK_PTR_NULL(out.mem[i]->ptr());
        out.ptr[i] = out.mem[i]->ptr();
        out.sz[i] = out.mem[i]->size();
        out.kind[i] = static_cast<u32>(k);
    }
    return HCCL_SUCCESS;
}

HcclResult PushDev(uint32_t n, const KHost &in, KDev &out)
{
    out.data = hccl::DeviceMem::alloc(n * sizeof(void *));
    CHK_PTR_NULL(out.data.ptr());
    out.size = hccl::DeviceMem::alloc(n * sizeof(u64));
    CHK_PTR_NULL(out.size.ptr());
    out.type = hccl::DeviceMem::alloc(n * sizeof(u32));
    CHK_PTR_NULL(out.type.ptr());

    CHK_RET(hrtMemSyncCopy(out.data.ptr(), n * sizeof(void *), in.ptr.data(), n * sizeof(void *),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    CHK_RET(hrtMemSyncCopy(out.size.ptr(), n * sizeof(u64), in.sz.data(), n * sizeof(u64),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    CHK_RET(hrtMemSyncCopy(out.type.ptr(), n * sizeof(u32), in.kind.data(), n * sizeof(u32),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

}  // namespace

HcclResult ChannelProcess::LaunchCommonChannelKernel(ChannelHandle *channelHandles,
    ChannelHandle *hostChannelHandles, uint32_t listNum, HcommChannelKind channelKind, aclrtBinHandle binHandle)
{
    HCCL_RUN_INFO("[%s] listNum[%u] HcommChannelRes path", __func__, listNum);

    KHost host;
    CHK_RET(PackHost(hostChannelHandles, listNum, channelKind, host));
    KDev dev;
    CHK_RET(PushDev(listNum, host, dev));

    hccl::DeviceMem deviceChannelList = hccl::DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());

    HcommChannelRes channelParam{};
    channelParam.channelList = static_cast<void *>(deviceChannelList.ptr());
    channelParam.listNum = listNum;
    channelParam.channelDataListAddr = static_cast<void *>(dev.data.ptr());
    channelParam.channelDataSizeListAddr = static_cast<void *>(dev.size.ptr());
    channelParam.channelTypeListAddr = static_cast<void *>(dev.type.ptr());
    CHK_RET(hrtGetDevice(&channelParam.deviceInfo.deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(channelParam.deviceInfo.deviceLogicId), channelParam.deviceInfo.devicePhyId));
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    channelParam.deviceInfo.deviceType = static_cast<u32>(devType);

    CHK_RET(LaunchKernelDeviceParam(channelParam, binHandle, "RunAicpuChannelInitV3"));

    CHK_RET(hrtMemSyncCopy(channelHandles,
        listNum * sizeof(ChannelHandle),
        deviceChannelList.ptr(),
        listNum * sizeof(ChannelHandle),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    CHK_RET(FillChannelD2HMap(channelHandles, hostChannelHandles, listNum));
    HCCL_INFO("[%s] channel kernel (HcommChannelRes) launch success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::LaunchChannelKernel(ChannelHandle *channelHandles,
    ChannelHandle *hostChannelHandles, HcommChannelDesc *hcommDesc, uint32_t listNum, aclrtBinHandle binHandle)
{
    HCCL_RUN_INFO("[%s] listNum[%u]", __func__, listNum);
    CHK_PRT_RET(listNum == 0U, HCCL_ERROR("[%s] listNum is 0", __func__), HCCL_E_PARA);
    auto *ch = reinterpret_cast<Channel *>(hostChannelHandles[0]);
    CHK_PTR_NULL(ch);
    if (ch->GetChannelKind() == HcommChannelKind::AICPU_TS_URMA
        || ch->GetChannelKind() == HcommChannelKind::AICPU_TS_UBOE) {
        return ChannelKernelLaunchForBase(channelHandles, hostChannelHandles, hcommDesc, listNum, binHandle);
    }
    return LaunchCommonChannelKernel(channelHandles, hostChannelHandles, listNum, ch->GetChannelKind(), binHandle);
}

HcclResult ChannelProcess::SaveChannels(ChannelHandle* targetChannels, ChannelHandle* userChannels,
    HcommChannelDesc *channelDescs, uint32_t channelNum, CommEngine engine, aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(targetChannels);
    CHK_PTR_NULL(userChannels);
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);

    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        if (channelNum == 0U) {
            return HCCL_SUCCESS;
        }
        CHK_RET(LaunchChannelKernel(userChannels, targetChannels, channelDescs, channelNum, binHandle));
    } else {
        HCCL_INFO("[%s] engine[%d] no need to KernelLaunch.", __func__, engine);
        for (uint32_t i = 0; i < channelNum; i++) {
            userChannels[i] = targetChannels[i];
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t *notifyNum)
{
    return WithChannelByHandleLocked(channelHandle, [&](Channel &channel) -> HcclResult {
        // 锁内调用，避免 destroy 并发释放
        channel.GetNotifyNum(notifyNum);
        return HcclResult::HCCL_SUCCESS;
    });
}

HcclResult ChannelProcess::ChannelGetRemoteMem(ChannelHandle channelHandle, CommMem **remoteMem, uint32_t *memNum, char **memTags)
{
    HcclMem **remoteMemConverted = reinterpret_cast<HcclMem **>(remoteMem);

    return WithChannelByHandleLocked(channelHandle, [&](Channel &channel) -> HcclResult {
        // 锁内调用，避免 destroy 并发释放
        channel.GetRemoteMem(remoteMemConverted, memNum, memTags);
        return HcclResult::HCCL_SUCCESS;
    });
}

HcclResult ChannelProcess::ChannelGetUserRemoteMem(ChannelHandle channelHandle, CommMem **remoteMem, char ***memTag, uint32_t *memNum)
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memTag);
    CHK_PTR_NULL(memNum);

    return WithChannelByHandleLocked(channelHandle, [&](Channel &channel) -> HcclResult {
        // 锁内调用，避免 destroy 并发释放
        channel.GetUserRemoteMem(remoteMem, memTag, memNum);
        return HcclResult::HCCL_SUCCESS;
    });
}

HcclResult ChannelProcess::ChannelGet(const ChannelHandle channelHandle, void **channel)
{
    CHK_PTR_NULL(channel);
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
    DeviceChannelKey key{deviceId, channelHandle};
    const auto &D2HhandleIter = g_ChannelD2HMap.find(key);
    if (D2HhandleIter == g_ChannelD2HMap.end()) {
        HCCL_ERROR("[ChannelProcess][%s] deviceId[%d], channel[%llx] not found.", __func__, deviceId, channelHandle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
 
    const auto handle = D2HhandleIter->second;
    const auto &handleIter = g_ChannelMap.find(handle);
    if (handleIter == g_ChannelMap.end()) {
        HCCL_ERROR("[ChannelProcess][%s] deviceId[%d], channel[%llx] not found.", __func__, deviceId, handle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    *channel = reinterpret_cast<void*>(handleIter->second.get());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelKernelDestroy(ChannelHandle *channelHandles, uint32_t listNum, aclrtBinHandle binHandle)
{
    HCCL_RUN_INFO("[%s] listNum[%u]", __func__, listNum);
    HcclChannelUrmaRes channelParam{};
    CHK_SAFETY_FUNC_RET(memset_s(&channelParam, sizeof(channelParam), 0, sizeof(channelParam)));

    // 将 host 侧的 channel handles 拷贝到 device 内存，供内核使用
    hccl::DeviceMem deviceChannelList = hccl::DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());
    CHK_RET(hrtMemSyncCopy(deviceChannelList.ptr(),
        listNum * sizeof(ChannelHandle),
        channelHandles,
        listNum * sizeof(ChannelHandle),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // 填充 channelParam（只需 channelList 和 listNum）
    channelParam.channelList = static_cast<void *>(deviceChannelList.ptr());
    channelParam.listNum = listNum;

    // 下 kernel
    std::string kernelName = "RunAicpuChannelDestroyV2";

    // 调用抽离的通用内核启动函数
    CHK_RET(LaunchKernel(channelParam, binHandle, kernelName));

    HCCL_INFO("[%s] channel kernel destroy success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::RemoveSingleChannel(int32_t deviceId, ChannelHandle inHandle,
    std::vector<ChannelHandle> &deviceHandles)
{
    DeviceChannelKey key{deviceId, inHandle};
    auto itH = g_ChannelD2HMap.find(key);
    if (itH == g_ChannelD2HMap.end()) {
        HCCL_ERROR("[Hcomm][%s] failed to find handle mapping in g_ChannelD2HMap, deviceId[%d], inHandle[0x%llx].",
            __func__, deviceId, inHandle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    const ChannelHandle mappedHandle = itH->second;

    auto itC = g_ChannelMap.find(mappedHandle);
    if (itC == g_ChannelMap.end()) {
        HCCL_ERROR("[Hcomm][%s] failed to find channel in g_ChannelMap, deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx].",
            __func__, deviceId, inHandle, mappedHandle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    deviceHandles.push_back(inHandle);

    HCCL_INFO("[Hcomm][%s] erase channel: deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx], ptr[%p]",
        __func__, deviceId, inHandle, mappedHandle, itC->second.get());

    g_ChannelMap.erase(itC);

    for (auto it = g_ChannelD2HMap.begin(); it != g_ChannelD2HMap.end();) {
        if (it->first.deviceId == deviceId && it->second == mappedHandle) {
            it = g_ChannelD2HMap.erase(it);
        } else {
            ++it;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelDestroy(const ChannelHandle *channels, uint32_t channelNum, aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(channels);
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum[0]", __func__), HCCL_E_PARA);
    HCCL_INFO("[%s] START. channelNum[%u].", __func__, channelNum);

    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::vector<ChannelHandle> deviceHandles;

    {
        std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
        for (uint32_t i = 0; i < channelNum; ++i) {
            HcclResult ret = RemoveSingleChannel(deviceId, channels[i], deviceHandles);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }
    }

    if (!deviceHandles.empty() && binHandle) {
        CHK_RET(ChannelKernelDestroy(deviceHandles.data(), deviceHandles.size(), binHandle));
    }
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelClean(const ChannelHandle *channelList, uint32_t listNum)
{
    CHK_PTR_NULL(channelList);

    for (uint32_t i = 0; i < listNum; ++i) {
        const ChannelHandle inHandle = channelList[i];
        // 单锁：D2H 映射 + 查 map + 锁内调用 Clean()
        HcclResult ret = WithChannelByHandleLocked(inHandle, [&](Channel &channel) -> HcclResult {
            return channel.Clean();  
        });
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[%s] ChannelHandle Clean failed, ret = 0x%016llx, i = %u", __func__, HCCL_ERROR_CODE(ret), i);
            return ret;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelResumeConcurrency(const ChannelHandle *channelList, uint32_t channelNum)
{
    for (uint32_t i = 0; i < channelNum; ++i) {
        const ChannelHandle inHandle = channelList[i];
        HcclResult ret = WithChannelByHandleLocked(inHandle, [&](Channel &channel) -> HcclResult {
            return channel.Resume();
        });
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[%s] Get ChannelHandle failed, ret = 0x%016llx, i = %u", __func__, HCCL_ERROR_CODE(ret), i);
            return ret;
        }
    }
    return HCCL_SUCCESS;
}
HcclResult ChannelProcess::ChannelResume(const ChannelHandle *channelList, uint32_t channelNum)
{
    CHK_PTR_NULL(channelList);

    // 1.resume resource
    HcclResult ret = ChannelResumeConcurrency(channelList, channelNum);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("HcommChannelResumeConcurrency error, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    HCCL_INFO("[%s] start resuming channels, timeout[%u]sec", __func__, timeout);

    std::vector<int32_t> statusVec(channelNum, 0);
    int32_t* statusList = statusVec.data();
    uint32_t retryCount{0};
    while (true) {
        HcclResult ret = ChannelGetStatus(channelList, channelNum, statusList);
        // 1. 检查超时
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            HCCL_ERROR("[%s] channel resume timeout after %u sec, channelNum[%u], elapsed[%lld]ms, retryCount[%u]",
                __func__, timeout, channelNum, elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }

        // 2. 处理重试（去除频繁的重试日志，一秒可能重试上千次）
        if (ret == HCCL_E_AGAIN) {
            ++retryCount;
            continue;
        }

        // 3. 处理失败
        if (ret != HCCL_SUCCESS) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            HCCL_ERROR("[%s] channel connect failed, channelNum[%u], ret[%d], elapsed[%lld]ms, retryCount[%u]",
                __func__, channelNum, ret, elapsed, retryCount);
            return ret;
        }

        // 4. 正常情况：所有通道连接成功
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime).count();
        HCCL_INFO("[%s] all channels connected successfully, channelNum[%u], elapsed[%lld]ms, retryCount[%u]",
            __func__, channelNum, elapsed, retryCount);
        break;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelUpdateKernelLaunch(ChannelHandle* deviceChannelHandles, ChannelHandle* hostChannelHandles, uint32_t listNum,
    const std::string &commTag, aclrtBinHandle binHandle)
{
    HCCL_RUN_INFO("[%s] listNum[%u], commTag[%s]", __func__, listNum, commTag.c_str());
    std::vector<std::vector<char>> hostPackBuffers(listNum);
    HcclChannelUrmaRes channelParam{};
    CHK_SAFETY_FUNC_RET(memset_s(&channelParam, sizeof(channelParam), 0, sizeof(channelParam)));

    // 获取host侧序列化的地址
    uint32_t totalListNum = 0;
    std::vector<u32> channelSizeVec{};
    for (uint32_t index = 0; index < listNum; index++) {
        auto aicpuTsUrmaChannel = reinterpret_cast<AicpuTsUrmaChannel *>(hostChannelHandles[index]);
        CHK_PRT(aicpuTsUrmaChannel->H2DResPack(hostPackBuffers[index]));   // todo:后续只打包connction
        totalListNum += hostPackBuffers[index].size();
        channelSizeVec.push_back(hostPackBuffers[index].size());
    }
    HCCL_INFO("[%s] totalListNum[%llu]", __func__, totalListNum);

    hccl::DeviceMem channelSizeAddr = hccl::DeviceMem::alloc(channelSizeVec.size() * sizeof(u32));
    CHK_PTR_NULL(channelSizeAddr.ptr());

    CHK_RET(hrtMemSyncCopy(channelSizeAddr.ptr(),
    channelSizeVec.size() * sizeof(u32),
    channelSizeVec.data(),
    channelSizeVec.size() * sizeof(u32),
    HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // 分配连续的host内存，将序列化的地址放入其中
    hccl::HostMem hostPackBuf = hccl::HostMem::alloc(totalListNum);
    CHK_PTR_NULL(hostPackBuf.ptr());
    CHK_RET(CombineHostMemory(hostPackBuffers, hostPackBuf));
    hccl::DeviceMem devicePackBuf = hccl::DeviceMem::alloc(totalListNum);
    CHK_PTR_NULL(devicePackBuf.ptr());

    // 将host侧序列化内容拷贝到device侧内存中
    CHK_RET(hrtMemSyncCopy(devicePackBuf.ptr(),
        totalListNum,
        hostPackBuf.ptr(),
        totalListNum,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // 填充channelParam参数
    s32 sRet = strncpy_s(channelParam.hcomId, HCOMID_MAX_LENGTH, commTag.c_str(), HCOMID_MAX_LENGTH - 1);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] str copy fail. return[%d]", __func__, sRet), HCCL_E_INTERNAL);
    channelParam.listNum = listNum;
    channelParam.uniqueIdAddr = static_cast<void *>(devicePackBuf.ptr());
    channelParam.uniqueIdSize = totalListNum;
    channelParam.channelSizeAddr = static_cast<void *>(channelSizeAddr.ptr());

    // 将 host 侧的 channel handles 拷贝到 device 内存，供内核使用
    hccl::DeviceMem deviceChannelList = hccl::DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());
    CHK_RET(hrtMemSyncCopy(deviceChannelList.ptr(),
        listNum * sizeof(ChannelHandle),
        deviceChannelHandles,
        listNum * sizeof(ChannelHandle),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    channelParam.channelList = static_cast<void *>(deviceChannelList.ptr());

    // 调用抽离的通用内核启动函数
    std::string kernelName = "RunAicpuIndOpChannelUpdateV2";
    CHK_RET(LaunchKernel(channelParam, binHandle, kernelName));

    HCCL_INFO("[%s] channel kernel launch success.", __func__);
    return HCCL_SUCCESS;
}

}
