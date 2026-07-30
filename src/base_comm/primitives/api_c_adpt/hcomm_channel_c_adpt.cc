/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <vector>

#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "hcomm_res.h"
#include "hcomm_result_defs.h"
#include "hcomm_res_defs.h"
#include "log.h"
#include "param_check_pub.h"
#include "comm_engine_utils.h"
#include "channel_process.h"
#include "aicpu_ts_channel_helper.h"
#ifdef ENABLE_EXPERIMENTAL
#include "nic_plugin_dispatcher.h"
#endif

using namespace hcomm;

HcommResult CheckUbAttr(HcommChannelDesc &channelDesc)
{
    if (channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBC_TP
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBOE
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBG
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBC_CTP) {
        return HCCL_SUCCESS;
    }

    // check sqDepth
    if (channelDesc.ubAttr.sqDepth == 0xFFFFFFFF) { // 0xFFFFFFFF表示使用默认值
        HCCL_INFO("[%s] use default ubAttr.sqDepth.", __func__);
        return HCCL_SUCCESS;
    }

    // sqDepth的合理范围在[16, 256]
    if (channelDesc.ubAttr.sqDepth < 16 || channelDesc.ubAttr.sqDepth > 256) {
        HCCL_ERROR(
            "[%s] invalid ubAttr.sqDepth[%u], should be 0 or >= 16 and <= 256.", __func__, channelDesc.ubAttr.sqDepth);
        return HCCL_E_PARA;
    }

    // channelDesc.ubAttr.sqDepth调整到2的整数次幂
    auto GetNextPowerOfTwo = [](uint32_t n) -> uint32_t {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    };

    channelDesc.ubAttr.sqDepth = GetNextPowerOfTwo(channelDesc.ubAttr.sqDepth);

    return HCCL_SUCCESS;
}

HcommResult CheckRoceAttr(HcommChannelDesc &channelDesc)
{
    if (channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_ROCE) {
        return HCCL_SUCCESS;
    }

    if (channelDesc.roceAttr.queueNum == INVALID_UINT) {
        channelDesc.roceAttr.queueNum = 1;
        HCCL_INFO("[%s] set roceAttr.queueNum to 1.", __func__);
    }

    return HCCL_SUCCESS;
}

namespace {
void ApplyHcommChannelDescV1Fields(const HcommChannelDesc &channelDesc, HcommChannelDesc &channelDescFinal)
{
    if (channelDesc.header.version < HCOMM_CHANNEL_VERSION_ONE) {
        return;
    }

    channelDescFinal.remoteEndpoint = channelDesc.remoteEndpoint;
    channelDescFinal.notifyNum = channelDesc.notifyNum;
    channelDescFinal.exchangeAllMems = channelDesc.exchangeAllMems;
    channelDescFinal.memHandles = channelDesc.memHandles;
    channelDescFinal.memHandleNum = channelDesc.memHandleNum;
    channelDescFinal.socket = channelDesc.socket;
    channelDescFinal.role = channelDesc.role;
    channelDescFinal.port = channelDesc.port;
}

HcommResult ProcessHcommChannelDescs(const HcommChannelDesc &channelDesc, HcommChannelDesc &channelDescFinal)
{
    if (channelDesc.header.size < sizeof(CommAbiHeader)) {
        HCCL_ERROR("[%s] invalid channelDesc.header.size[%u].", __func__, channelDesc.header.size);
        return HCCL_E_PARA;
    }

    if (channelDesc.header.magicWord != channelDescFinal.header.magicWord) {
        HCCL_ERROR("[%s] channelDesc.header.magicWord[0x%08x] is invalid, expected[0x%08x].", __func__,
            channelDesc.header.magicWord, channelDescFinal.header.magicWord);
        return HCCL_E_PARA;
    }

    const uint32_t copySize = (channelDescFinal.header.size < channelDesc.header.size ? channelDescFinal.header.size
                                                                                      : channelDesc.header.size)
                              - sizeof(CommAbiHeader);
    CHK_SAFETY_FUNC_RET(memcpy_s(reinterpret_cast<uint8_t *>(&channelDescFinal) + sizeof(CommAbiHeader), copySize,
        reinterpret_cast<const uint8_t *>(&channelDesc) + sizeof(CommAbiHeader), copySize));
    ApplyHcommChannelDescV1Fields(channelDesc, channelDescFinal);
    if (channelDesc.header.version > HCOMM_CHANNEL_VERSION) {
        HCCL_RUN_WARNING("The version of provided [%u] is higher than the current version[%u], "
                         "unsupported configuration will be ignored.",
            channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    } else if (channelDesc.header.version < HCOMM_CHANNEL_VERSION) {
        HCCL_RUN_WARNING("The version of provided [%u] is lower than the current version[%u], "
                         "configurations supported by later versions will be ignored.",
            channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    }

    // qos：低版本时置默认值
    if (channelDesc.header.version <= HCOMM_CHANNEL_VERSION_ONE) {
        channelDescFinal.qos = 0xFFFFFFFFU;
    } else {
        channelDescFinal.qos = channelDesc.qos;
    }

    // v3：channelName，低版本时置 NULL
    if (channelDesc.header.version < HCOMM_CHANNEL_VERSION) {
        channelDescFinal.channelName = nullptr;
    } else {
        channelDescFinal.channelName = channelDesc.channelName;
        if (channelDescFinal.channelName != nullptr
            && reinterpret_cast<uintptr_t>(channelDescFinal.channelName) == static_cast<uintptr_t>(-1)) {
            channelDescFinal.channelName = nullptr;
        }
    }

    if (channelDescFinal.channelName != nullptr) {
        size_t nameLen = strnlen(channelDescFinal.channelName, HCOMM_CHANNEL_NAME_MAX_LEN + 1);
        if (nameLen > HCOMM_CHANNEL_NAME_MAX_LEN) {
            HCCL_ERROR("[%s] channelName too long, max len[%u].", __func__, HCOMM_CHANNEL_NAME_MAX_LEN);
            return HCCL_E_PARA;
        }
    }

    return HCOMM_SUCCESS;
}

HcommResult NormalizeHcommChannelDescs(
    HcommChannelDesc *channelDescs, uint32_t channelNum, std::vector<HcommChannelDesc> &channelDescFinals)
{
    channelDescFinals.clear();
    channelDescFinals.reserve(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        HcommChannelDesc channelDescFinal{};
        HcommResult ret = HcommChannelDescInit(&channelDescFinal, 1);
        if (ret != HCOMM_SUCCESS) {
            return ret;
        }
        ret = ProcessHcommChannelDescs(channelDescs[idx], channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] failed to normalize channelDesc[%u], ret[%d].", __func__, idx, ret);
            return ret;
        }
        ret = CheckUbAttr(channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] CheckUbAttr failed, ret[%d].", __func__, ret);
            return ret;
        }
        ret = CheckRoceAttr(channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] CheckRoceAttr failed, ret[%d].", __func__, ret);
            return ret;
        }

        channelDescFinals.push_back(channelDescFinal);
    }
    return HCOMM_SUCCESS;
}
} // namespace

// 集合通信使用，待归一到HcommChannelCreate
HcommResult HcommCollectiveChannelCreate(EndpointHandle endpointHandle, CommEngine engine,
    HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u].", __func__, endpointHandle,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum);

    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(NormalizeHcommChannelDescs(channelDescs, channelNum, channelDescFinals)));
    return ChannelProcess::CreateChannelsLoop(endpointHandle, engine, channelDescFinals.data(), channelNum, channels);
}

HcommResult HcommChannelUpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum, ChannelHandle channelHandle)
{
    CHK_PTR_NULL(memHandles);
    CHK_PRT_RET((memHandleNum == 0), HCCL_ERROR("[%s]Invalid memHandleNum, memHandleNum is 0.", __func__), HCCL_E_PARA);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelUpdateMemInfo(channelHandle, memHandles, memHandleNum, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    return ChannelProcess::ChannelUpdateMemInfo(memHandles, memHandleNum, channelHandle);
}

HcommResult HcommChannelCreate(EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc *channelDescs,
    uint32_t channelNum, ChannelHandle *channels)
{
    CHK_PTR_NULL(endpointHandle);
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u].", __func__, endpointHandle,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum);
    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    if (endpoint != nullptr) {
        CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    }
    (void)HcommResMgrInit();
    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(NormalizeHcommChannelDescs(channelDescs, channelNum, channelDescFinals)));

#ifdef ENABLE_EXPERIMENTAL
    bool pluginHandled = false;
    CHK_RET(static_cast<HcclResult>(
        PluginChannelCreate(endpointHandle, engine, channelDescFinals.data(), channelNum, channels, pluginHandled)));
    if (pluginHandled) {
        return HCCL_SUCCESS;
    }
#endif

    std::vector<ChannelHandle> hostChannelHandles(channelNum);
    ChannelHandle *targetChannels = hostChannelHandles.data();

    CHK_RET(ChannelProcess::CreateChannelsLoop(
        endpointHandle, engine, channelDescFinals.data(), channelNum, targetChannels));
    CHK_RET(
        ChannelProcess::PrepareUserChannels(targetChannels, channels, channelDescFinals.data(), channelNum, engine));

    return HCCL_SUCCESS;
}

HcommResult HcommChannelGet(ChannelHandle channelHandle, void **channel)
{
    CHK_PTR_NULL(channel);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelGet(channelHandle, channel, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif
    return ChannelProcess::ChannelGet(channelHandle, channel);
}

HcommResult HcommChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList)
{
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(statusList);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s]Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);
    (void)HcommResMgrInit();
#ifdef ENABLE_EXPERIMENTAL
    bool allHandled = true;
    for (uint32_t i = 0; i < listNum; i++) {
        bool handled = false;
        CHK_RET(static_cast<HcclResult>(PluginChannelGetStatus(channelList[i], &statusList[i], handled)));
        if (!handled) {
            allHandled = false;
        }
    }
    if (allHandled) {
        return HCCL_SUCCESS;
    }
#endif

    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> channelDescFinals;
    std::vector<ChannelStatus> internalStatus(listNum);
    HcclResult ret = ChannelProcess::GetChannelsInfo(channelList, listNum, engines, channelDescFinals, internalStatus);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] GetChannelsInfo failed, ret[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    ret = ChannelProcess::HandleStatusByEngine(
        channelList, listNum, engines, channelDescFinals, internalStatus, statusList);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] HandleStatusByEngine failed, ret[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t *notifyNum)
{
    CHK_PTR_NULL(notifyNum);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelGetNotifyNum(channelHandle, notifyNum, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif
    return ChannelProcess::ChannelGetNotifyNum(channelHandle, notifyNum);
}

HcommResult HcommChannelDestroy(const ChannelHandle *channels, uint32_t channelNum)
{
    CHK_PTR_NULL(channels);
    (void)HcommResMgrInit();
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    std::vector<ChannelHandle> builtinChannels;
    builtinChannels.reserve(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
#ifdef ENABLE_EXPERIMENTAL
        bool handled = false;
        CHK_RET(static_cast<HcclResult>(PluginChannelDestroy(channels[idx], handled)));
        if (handled) {
            continue;
        }
#endif
        builtinChannels.push_back(channels[idx]);
    }
    if (builtinChannels.empty()) {
        return HCCL_SUCCESS;
    }
    return ChannelProcess::ChannelDestroy(
        builtinChannels.data(), builtinChannels.size(), AicpuTsChannelHelper::GetBinHandle());
}

HcommResult HcommChannelGetRemoteMems(
    ChannelHandle channelHandle, uint32_t *memNum, CommMem **remoteMem, char ***memInfos)
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memNum);
    CHK_PTR_NULL(memInfos);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelGetRemoteMems(channelHandle, memNum, remoteMem, memInfos, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    return ChannelProcess::ChannelGetRemoteMems(channelHandle, memNum, remoteMem, memInfos);
}
