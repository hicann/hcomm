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
#include "channel_config.h"
#include "shared_jetty_mgr.h"
#include "endpoint.h"

using namespace hcomm;

HcommResult CheckUbAttr(HcommChannelDesc& channelDesc)
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

HcommResult CheckRoceAttr(HcommChannelDesc& channelDesc)
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
void ApplyHcommChannelDescV1Fields(const HcommChannelDesc& channelDesc, HcommChannelDesc& channelDescFinal)
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

HcommResult ProcessHcommChannelDescs(const HcommChannelDesc& channelDesc, HcommChannelDesc& channelDescFinal)
{
    if (channelDesc.header.size < sizeof(CommAbiHeader)) {
        HCCL_ERROR("[%s] invalid channelDesc.header.size[%u].", __func__, channelDesc.header.size);
        return HCCL_E_PARA;
    }

    if (channelDesc.header.magicWord != channelDescFinal.header.magicWord) {
        HCCL_ERROR(
            "[%s] channelDesc.header.magicWord[0x%08x] is invalid, expected[0x%08x].", __func__,
            channelDesc.header.magicWord, channelDescFinal.header.magicWord);
        return HCCL_E_PARA;
    }

    const uint32_t copySize = (channelDescFinal.header.size < channelDesc.header.size ? channelDescFinal.header.size :
                                                                                        channelDesc.header.size)
                              - sizeof(CommAbiHeader);
    CHK_SAFETY_FUNC_RET(memcpy_s(
        reinterpret_cast<uint8_t*>(&channelDescFinal) + sizeof(CommAbiHeader), copySize,
        reinterpret_cast<const uint8_t*>(&channelDesc) + sizeof(CommAbiHeader), copySize));
    ApplyHcommChannelDescV1Fields(channelDesc, channelDescFinal);
    if (channelDesc.header.version > HCOMM_CHANNEL_VERSION) {
        HCCL_RUN_WARNING(
            "The version of provided [%u] is higher than the current version[%u], "
            "unsupported configuration will be ignored.",
            channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    } else if (channelDesc.header.version < HCOMM_CHANNEL_VERSION) {
        HCCL_RUN_WARNING(
            "The version of provided [%u] is lower than the current version[%u], "
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
    HcommChannelDesc* channelDescs, uint32_t channelNum, std::vector<HcommChannelDesc>& channelDescFinals)
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
HcommResult HcommCollectiveChannelCreate(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    HCCL_INFO(
        "[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u].", __func__, endpointHandle,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum);

    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(NormalizeHcommChannelDescs(channelDescs, channelNum, channelDescFinals)));
    return ChannelProcess::CreateChannelsLoop(endpointHandle, engine, channelDescFinals.data(), channelNum, channels);
}

HcommResult HcommChannelUpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum, ChannelHandle channelHandle)
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

HcommResult HcommChannelCreate(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels)
{
    CHK_PTR_NULL(endpointHandle);
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    HCCL_INFO(
        "[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u].", __func__, endpointHandle,
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
    ChannelHandle* targetChannels = hostChannelHandles.data();

    CHK_RET(ChannelProcess::CreateChannelsLoop(
        endpointHandle, engine, channelDescFinals.data(), channelNum, targetChannels));
    CHK_RET(
        ChannelProcess::PrepareUserChannels(targetChannels, channels, channelDescFinals.data(), channelNum, engine));

    return HCCL_SUCCESS;
}

HcommResult HcommChannelGet(ChannelHandle channelHandle, void** channel)
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

HcommResult HcommChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList)
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

HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t* notifyNum)
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

#ifdef ENABLE_EXPERIMENTAL
static void UnregisterPluginChannels(std::vector<ChannelHandle>& pluginChannels)
{
    // plugin channel 虽由 PluginChannelDestroy 销毁，但创建时同样经 RegisterChannels 注册，
    // 需在此统一注销，否则 CheckEndpointDestroy 会因残留记录阻止 Endpoint 销毁。
    // 无论 PluginChannelDestroy 是否失败，已收集的 pluginChannels 都需注销。
    if (!pluginChannels.empty()) {
        (void)hcomm::SharedJettyMgr::GetInstance().UnregisterChannels(pluginChannels.data(), pluginChannels.size());
    }
}
#endif

static HcclResult DestroyBuiltinChannels(std::vector<ChannelHandle>& builtinChannels)
{
    // 即使 plugin channel 销毁失败，也需继续销毁 builtin channel，避免 RDMA/jetty 资源泄漏
    // 及 SharedJettyMgr 残留记录永久阻塞 Endpoint 销毁。最终返回首个错误（优先 plugin 错误）。
    HcclResult builtinRet = HCCL_SUCCESS;
    if (builtinChannels.empty()) {
        return builtinRet;
    }
    builtinRet = ChannelProcess::ChannelDestroy(
        builtinChannels.data(), builtinChannels.size(), AicpuTsChannelHelper::GetBinHandle());
    // 无论 ChannelDestroy 成功与否都注销 SharedJettyMgr 记录：
    // 成功时正常清理；失败时 channel 已不可用，若不注销会永久阻塞 Endpoint 销毁。
    if (builtinRet != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[%s] ChannelDestroy failed, ret[%d], force unregister shared jetty channels.", __func__, builtinRet);
    }
    (void)hcomm::SharedJettyMgr::GetInstance().UnregisterChannels(builtinChannels.data(), builtinChannels.size());
    return builtinRet;
}

HcommResult HcommChannelDestroy(const ChannelHandle* channels, uint32_t channelNum)
{
    CHK_PTR_NULL(channels);
    (void)HcommResMgrInit();
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    std::vector<ChannelHandle> builtinChannels;
    builtinChannels.reserve(channelNum);
#ifdef ENABLE_EXPERIMENTAL
    std::vector<ChannelHandle> pluginChannels;
    pluginChannels.reserve(channelNum);
    HcclResult pluginRet = HCCL_SUCCESS;
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        bool handled = false;
        HcclResult curRet = static_cast<HcclResult>(PluginChannelDestroy(channels[idx], handled));
        if (curRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] PluginChannelDestroy failed, idx[%u], ret[%d], handled[%d].", __func__, idx, curRet,
                static_cast<int>(handled));
            pluginRet = curRet;
            // 不 break，继续处理剩余 channel，确保全部得到销毁/注销
            if (handled) {
                // plugin 已认领 channel 所有权，即使返回错误也由 plugin 负责清理，
                // 不再走 builtin 路径以免 double-free，但仍需注销 SharedJettyMgr 记录
                pluginChannels.push_back(channels[idx]);
                continue;
            }
            // handled=false 契约保证 plugin 未修改 channel 状态且应返回 HCCL_SUCCESS；
            // 此分支为防御性兜底（当前实现不可达），仍按 builtin 路径销毁避免资源泄漏。
            HCCL_WARNING(
                "[%s] PluginChannelDestroy returned error with handled=false, idx[%u], ret[%d]. "
                "Falling back to builtin destroy per contract.",
                __func__, idx, curRet);
        } else if (handled) {
            pluginChannels.push_back(channels[idx]);
            continue;
        }
        // 非 plugin channel 或 plugin 未认领，按 builtin 路径处理
        builtinChannels.push_back(channels[idx]);
    }
    UnregisterPluginChannels(pluginChannels);
    HcclResult builtinRet = DestroyBuiltinChannels(builtinChannels);
    if (pluginRet != HCCL_SUCCESS) {
        return pluginRet;
    }
    return builtinRet;
#else
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        builtinChannels.push_back(channels[idx]);
    }
    return static_cast<HcommResult>(DestroyBuiltinChannels(builtinChannels));
#endif
}

HcommResult HcommChannelConfigCreate(HcommChannelConfig* config)
{
    return static_cast<HcommResult>(hcomm::ChannelConfigCreate(config));
}

HcommResult HcommChannelConfigDestroy(HcommChannelConfig config)
{
    return static_cast<HcommResult>(hcomm::ChannelConfigDestroy(config));
}

HcommResult HcommChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value)
{
    return static_cast<HcommResult>(hcomm::ChannelConfigSetInt(config, type, value));
}

static bool IsUbProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP;
}

static HcclResult ValidateSharedQueueConfig(const std::vector<HcommChannelDesc>& channelDescs)
{
    for (uint32_t i = 0; i < channelDescs.size(); ++i) {
        CommProtocol protocol = channelDescs[i].remoteEndpoint.protocol;
        if (!IsUbProtocol(protocol)) {
            HCCL_ERROR(
                "[%s] IS_SHARED_QUEUE only supports UB protocols (UBC_CTP/UBC_TP), "
                "channelDesc[%u] protocol[%d].",
                __func__, i, protocol);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

#ifdef ENABLE_EXPERIMENTAL
static HcclResult
RegisterSharedQueuePluginChannels(EndpointHandle endpointHandle, uint32_t channelNum, ChannelHandle* channels)
{
    // plugin channel 不经过 Channel::BuildConnection，无法走共享 jetty 复用路径。
    // 但仍需注册到 SharedJettyMgr 以便 CheckEndpointDestroy 校验，确保 plugin channel
    // 销毁后 endpoint 才能销毁（plugin channel 与 endpoint 也有资源依赖关系）。
    HcclResult regRet = hcomm::SharedJettyMgr::GetInstance().RegisterChannels(endpointHandle, channels, channelNum);
    if (regRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] failed to register shared jetty channels (plugin), ret[%d].", __func__, regRet);
        // 注册失败需销毁已创建的 plugin channel，避免泄漏
        for (uint32_t i = 0; i < channelNum; ++i) {
            bool handled = false;
            (void)static_cast<HcclResult>(PluginChannelDestroy(channels[i], handled));
        }
        return regRet;
    }
    return HCCL_SUCCESS;
}
#endif

static HcclResult CreateAndRegisterSharedQueueBuiltinChannels(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescFinals, uint32_t channelNum,
    ChannelHandle* channels)
{
    // 共享模式建链流程与 HcommChannelCreate 一致：CreateChannelsLoop 传 isSharedQueue=true，
    // channel 的 BuildConnection 据此走共享 jetty 复用路径；PrepareUserChannels 完成 AICPU/AIV 预分配。
    std::vector<ChannelHandle> hostChannelHandles(channelNum);
    ChannelHandle* targetChannels = hostChannelHandles.data();

    CHK_RET(ChannelProcess::CreateChannelsLoop(
        endpointHandle, engine, channelDescFinals, channelNum, targetChannels, true));
    HcclResult prepRet
        = ChannelProcess::PrepareUserChannels(targetChannels, channels, channelDescFinals, channelNum, engine);
    if (prepRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] PrepareUserChannels failed, ret[%d], destroying created channels.", __func__, prepRet);
        (void)ChannelProcess::ChannelDestroy(targetChannels, channelNum, AicpuTsChannelHelper::GetBinHandle());
        return prepRet;
    }

    HcclResult regRet = hcomm::SharedJettyMgr::GetInstance().RegisterChannels(endpointHandle, channels, channelNum);
    if (regRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] failed to register shared jetty channels, ret[%d].", __func__, regRet);
        (void)ChannelProcess::ChannelDestroy(channels, channelNum, AicpuTsChannelHelper::GetBinHandle());
        return regRet;
    }
    return HCCL_SUCCESS;
}

HcommResult HcommChannelCreateWithConfig(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    HcommChannelConfig config, ChannelHandle* channels)
{
    CHK_PTR_NULL(endpointHandle);
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    HCCL_INFO(
        "[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u], config[%p].", __func__, endpointHandle,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, config);

    bool isSharedQueue = false;
    if (config != nullptr) {
        auto* cfg = static_cast<hcomm::HcommChannelConfigData*>(config);
        isSharedQueue = cfg->isSharedQueue;
    }

    // 非共享模式直接复用 HcommChannelCreate 流程，避免重复维护两套建链逻辑
    if (!isSharedQueue) {
        return HcommChannelCreate(endpointHandle, engine, channelDescs, channelNum, channels);
    }

    // 共享 jetty 仅支持 AIV 引擎：AICPU 等 channel 的 BuildConnection 不处理共享 jetty 路径，
    // 强行创建会导致 channel 注册到 SharedJettyMgr 但无实际 jetty 共享，多 channel 共用同一 SQ
    // 但 PI/CI 未协调，引发 WQE 覆盖、doorbell 不前进、notify 超时。
    if (engine != COMM_ENGINE_AIV) {
        HCCL_ERROR(
            "[%s] IS_SHARED_QUEUE currently only supports AIV engine, engine[%d].", __func__, static_cast<int>(engine));
        return HCCL_E_NOT_SUPPORT;
    }

    auto endpoint = GetEndpointMap().GetEndpoint(endpointHandle);
    if (endpoint != nullptr) {
        CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    }
    (void)HcommResMgrInit();

    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(NormalizeHcommChannelDescs(channelDescs, channelNum, channelDescFinals)));
    // NormalizeHcommChannelDescs 内部已调 CheckUbAttr，此处仅补共享模式专有校验
    CHK_RET(ValidateSharedQueueConfig(channelDescFinals));

#ifdef ENABLE_EXPERIMENTAL
    bool pluginHandled = false;
    CHK_RET(static_cast<HcclResult>(
        PluginChannelCreate(endpointHandle, engine, channelDescFinals.data(), channelNum, channels, pluginHandled)));
    if (pluginHandled) {
        return static_cast<HcommResult>(RegisterSharedQueuePluginChannels(endpointHandle, channelNum, channels));
    }
#endif

    HcclResult ret = CreateAndRegisterSharedQueueBuiltinChannels(
        endpointHandle, engine, channelDescFinals.data(), channelNum, channels);
    if (ret != HCCL_SUCCESS) {
        return static_cast<HcommResult>(ret);
    }

    HCCL_INFO("[%s] SUCCESS. isSharedQueue[%d], channelNum[%u].", __func__, isSharedQueue, channelNum);
    return HCCL_SUCCESS;
}

HcommResult
HcommChannelGetRemoteMems(ChannelHandle channelHandle, uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
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
