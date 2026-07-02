/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "my_rank.h"
#include <algorithm>
#include <limits>
#include "hccl_comm_pub.h"
#include "exception_handler.h"
#include "config_log.h"
#include "config/env_config.h"
#include "env_config/env_config.h"
#include "../common/loggers/channel_logger.h"  // 日志记录器

#include "hcom_common.h"
#include "ccu_kernel.h"
#include "ccu_kernel_mgr.h"
#include "rt_external.h"
#include "coll_comm_mgr.h"
#include "hcclCommOp.h"
#include "channel_process.h"
#include "aicpu_ts_roce_channel_v2.h"
#include "aiv_urma_channel.h"
#include "hccl_group.h"
#include "../resource_mgr/local/my_rank/comm_engine_reses/kernel_launch/hccl_kernel_launch_aicpu.h"
#include "param_check_basic_v2.h"
#include "comm_engine_utils.h"

using namespace hccl;
/**
 * @note 职责：集合通信的通信域资源管理的C接口的C到C++适配
 */

/**
 * @note C接口适配参考示例
 * @code {.c}
 * HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
 *     uint32_t notifyNumPerThread, ThreadHandle *threads) {
 *     return HCCL_SUCCESS;
 * }
 * @endcode
 */

constexpr uint32_t HCCL_CHANNEL_VERSION_ONE = 1;
constexpr uint32_t MULTIPLE = 4;               // 用于A5判断TC是否为4的倍数
constexpr uint32_t TC_MAX = 255;               // TC的最大值（不区分芯片类型）
constexpr uint32_t RETRY_INTERVAL_MIN = 5u;    // retryInterval范围的最小值（不区分芯片类型）
constexpr uint32_t A5_RETRY_INTERVAL_MAX = 24u;// A5的retryInterval范围的最大值
constexpr uint32_t RETRY_CNT_MIN = 1u;         // retryCnt范围的最小值（不区分芯片类型）
constexpr uint32_t RETRY_CNT_MAX = 7u;         // retryCnt范围的最大值（不区分芯片类型）
constexpr uint32_t SL_MAX = 7u;                // sl范围的最大值，sl即serviceLevel（不区分芯片类型）
constexpr uint32_t TC_DEFAULT = 0xFFFFFFFFu;   // TC的默认值（不区分芯片类型）
constexpr uint32_t SL_DEFAULT = 0xFFFFFFFFu;   // SL的默认值（不区分芯片类型）

static void FillChannelDescFinal(hccl::CommConfig commConfig, const HcclChannelDesc &channelDesc, HcclChannelDesc &channelDescFinal, bool isCommunicatorV2)
{
    if (isCommunicatorV2) { // A5
        auto& rdmaConfig = Hccl::EnvConfig::GetInstance().GetRdmaConfig();
        channelDescFinal.roceAttr.retryCnt = (channelDesc.roceAttr.retryCnt == INVALID_UINT) ? rdmaConfig.GetRdmaRetryCnt() : channelDesc.roceAttr.retryCnt;
        channelDescFinal.roceAttr.retryInterval = (channelDesc.roceAttr.retryInterval == INVALID_UINT) ? rdmaConfig.GetRdmaTimeOut() : channelDesc.roceAttr.retryInterval;
        channelDescFinal.roceAttr.tc = (commConfig.GetConfigTrafficClass() == INVALID_UINT) ? rdmaConfig.GetRdmaTrafficClass() : commConfig.GetConfigTrafficClass();
        channelDescFinal.roceAttr.sl = (commConfig.GetConfigServiceLevel() == INVALID_UINT) ? rdmaConfig.GetRdmaServerLevel() : commConfig.GetConfigServiceLevel();
        channelDescFinal.roceAttr.queueNum = (channelDesc.roceAttr.queueNum == INVALID_UINT) ? rdmaConfig.GetRdmaQueueNum() : channelDesc.roceAttr.queueNum;
    } else {
        channelDescFinal.roceAttr.retryCnt = (channelDesc.roceAttr.retryCnt == INVALID_UINT) ? EnvConfig::GetExternalInputRdmaRetryCnt() : channelDesc.roceAttr.retryCnt;
        channelDescFinal.roceAttr.retryInterval = (channelDesc.roceAttr.retryInterval == INVALID_UINT) ? EnvConfig::GetExternalInputRdmaTimeOut() : channelDesc.roceAttr.retryInterval;
        channelDescFinal.roceAttr.tc = (channelDesc.roceAttr.tc == 0xFF) ? EnvConfig::GetExternalInputRdmaTrafficClass() : channelDesc.roceAttr.tc;
        channelDescFinal.roceAttr.sl = (channelDesc.roceAttr.sl == 0xFF) ? EnvConfig::GetExternalInputRdmaServerLevel() : channelDesc.roceAttr.sl;
        channelDescFinal.roceAttr.queueNum = (channelDesc.roceAttr.queueNum == INVALID_UINT) ? GetExternalInputQpsPerConnection() : channelDesc.roceAttr.queueNum;
    }
}

static HcclResult CheckA5Config(hccl::CommConfig commConfig, const HcclChannelDesc &channelDesc)
{
    u32 tc = commConfig.GetConfigTrafficClass();
    CHK_PRT_RET((tc != TC_DEFAULT) && (tc > TC_MAX || (tc % MULTIPLE != 0)),
        HCCL_ERROR("[ProcessRoceChannelDesc]errNo[0x%016llx] invalid hcclRdmaTrafficClass[%u], must be 0xFFFFFFFF or in [0,255] and a multiple of 4",
            HCCL_ERROR_CODE(HCCL_E_PARA), tc),
        HCCL_E_PARA);

    u32 sl = commConfig.GetConfigServiceLevel();
    CHK_PRT_RET((sl != SL_DEFAULT) && (sl > SL_MAX),
        HCCL_ERROR("[ProcessRoceChannelDesc]errNo[0x%016llx] invalid hcclRdmaServiceLevel[%u], must be 0xFFFFFFFF or in [0,7]",
            HCCL_ERROR_CODE(HCCL_E_PARA), sl),
        HCCL_E_PARA);

    u32 retryInterval = channelDesc.roceAttr.retryInterval;
    CHK_PRT_RET((retryInterval != INVALID_UINT) && (retryInterval < RETRY_INTERVAL_MIN || retryInterval > A5_RETRY_INTERVAL_MAX),
        HCCL_ERROR("[ProcessRoceChannelDesc]errNo[0x%016llx] invalid hcclRdmaRetryInterval[%u], must be 0xFFFFFFFF or in [5,24]",
        HCCL_ERROR_CODE(HCCL_E_PARA), retryInterval),
        HCCL_E_PARA);

    u32 retryCnt = channelDesc.roceAttr.retryCnt;
    CHK_PRT_RET((retryCnt != INVALID_UINT) && (retryCnt < RETRY_CNT_MIN || retryCnt > RETRY_CNT_MAX),
        HCCL_ERROR("[ProcessRoceChannelDesc]errNo[0x%016llx] invalid hcclRdmaRetryCnt[%u], must be 0xFFFFFFFF or in [1,7]",
        HCCL_ERROR_CODE(HCCL_E_PARA), retryCnt),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult ProcessRoceChannelDesc(const HcclChannelDesc &channelDesc, HcclChannelDesc &channelDescFinal, hccl::hcclComm *hcclComm)
{
    bool isCommunicatorV2 = hcclComm->IsCommunicatorV2();
    hccl::CommConfig commConfig{}; // A5使用
    if (isCommunicatorV2) { // A5
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        commConfig = collComm->GetCommConfig();
        CHK_RET(CheckA5Config(commConfig, channelDesc));
    }
    FillChannelDescFinal(commConfig, channelDesc, channelDescFinal, isCommunicatorV2);
    HCCL_INFO("[%s]queueNum[%u], retryCnt[%u], retryInterval[%u], tc[%u], sl[%u]", __func__,
        channelDescFinal.roceAttr.queueNum, channelDescFinal.roceAttr.retryCnt, channelDescFinal.roceAttr.retryInterval,
        channelDescFinal.roceAttr.tc, channelDescFinal.roceAttr.sl);
    return HCCL_SUCCESS;
}

HcclResult ProcessUbChannelDesc(const HcclChannelDesc &channelDesc, HcclChannelDesc &channelDescFinal,
    hccl::hcclComm *hcclComm)
{
    (void)channelDescFinal;
    (void)hcclComm;

    if (channelDesc.channelProtocol != COMM_PROTOCOL_UBC_CTP &&
        channelDesc.channelProtocol != COMM_PROTOCOL_UBC_TP &&
        channelDesc.channelProtocol != COMM_PROTOCOL_UBOE) {
        HCCL_ERROR("[%s] unexpected channelProtocol[%d], expect UBC_CTP/UBC_TP/UBOE", __func__,
            static_cast<int>(channelDesc.channelProtocol));
        return HCCL_E_PARA;
    }
    HCCL_INFO("[%s] channelProtocol[%d] ub comm-domain qos applied in HcommChannelDesc::qos when converting (HcclChannelDesc has no qos field)",
        __func__, static_cast<int>(channelDesc.channelProtocol));
    return HCCL_SUCCESS;
}

HcclResult ProcessHcclChannelDesc(const HcclChannelDesc &channelDesc, HcclChannelDesc &channelDescFinal, hccl::hcclComm *hcclComm)
{
    channelDescFinal.remoteRank = channelDesc.remoteRank;
    channelDescFinal.channelProtocol   = channelDesc.channelProtocol;
    channelDescFinal.localEndpoint  = channelDesc.localEndpoint;
    channelDescFinal.remoteEndpoint  = channelDesc.remoteEndpoint;
    channelDescFinal.notifyNum  = channelDesc.notifyNum;
    channelDescFinal.memHandles  = channelDesc.memHandles;
    channelDescFinal.memHandleNum  = channelDesc.memHandleNum;

     // 根据协议类型拷贝union中的相应成员
    switch (channelDesc.channelProtocol) {
        case COMM_PROTOCOL_HCCS:
        case COMM_PROTOCOL_HCCS_ONLY:
        case COMM_PROTOCOL_PCIE:
        case COMM_PROTOCOL_SIO:
        case COMM_PROTOCOL_UB_MEM:
            break;
        case COMM_PROTOCOL_UBC_CTP:
        case COMM_PROTOCOL_UBC_TP:
        case COMM_PROTOCOL_UBOE:
            return ProcessUbChannelDesc(channelDesc, channelDescFinal, hcclComm);
        case COMM_PROTOCOL_ROCE:
            return ProcessRoceChannelDesc(channelDesc, channelDescFinal, hcclComm);
        default: {
            auto ProtocolToString = [](const CommProtocol proto) -> const char* {
                switch (proto) {
                    case COMM_PROTOCOL_HCCS:    return "COMM_PROTOCOL_HCCS";
                    case COMM_PROTOCOL_PCIE:    return "COMM_PROTOCOL_PCIE";
                    case COMM_PROTOCOL_SIO:     return "COMM_PROTOCOL_SIO";
                    case COMM_PROTOCOL_UBC_CTP: return "COMM_PROTOCOL_UBC_CTP";
                    case COMM_PROTOCOL_UB_MEM:  return "COMM_PROTOCOL_UB_MEM";
                    case COMM_PROTOCOL_ROCE:    return "COMM_PROTOCOL_ROCE";
                    case COMM_PROTOCOL_UBC_TP:  return "COMM_PROTOCOL_UBC_TP";
                    case COMM_PROTOCOL_UBOE:    return "COMM_PROTOCOL_UBOE";
                    case COMM_PROTOCOL_HCCS_ONLY:   return "COMM_PROTOCOL_HCCS_ONLY";
                    default:                    return "UNKNOWN_PROTOCOL";
                }
            };
            HCCL_ERROR("[%s] Unsupported protocol[%s] found in HcclChannelDesc.",
                       __func__, ProtocolToString(channelDesc.channelProtocol));
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProcessHcclResPackReq(const HcclChannelDesc &channelDesc, HcclChannelDesc &channelDescFinal, hccl::hcclComm *hcclComm)
{
    if (channelDesc.header.size < channelDescFinal.header.size) {
        // 需要前向兼容HcclChannelDesc，末尾部分字段不支持处理
    } else if (channelDesc.header.size > channelDescFinal.header.size) {
        // 需要后向向兼容HcclChannelDesc，末尾部分字段会被忽略
    }
 
    if (channelDesc.header.magicWord != channelDescFinal.header.magicWord) {
        HCCL_ERROR("[%s]channelDescFinal.header.magicWord[%u] not equal to channelDesc.header.magicWord[%u]",
            __func__, channelDescFinal.header.magicWord, channelDesc.header.magicWord);
        return HCCL_E_PARA;
    }
 
    uint32_t copySize = (channelDescFinal.header.size < channelDesc.header.size ?
        channelDescFinal.header.size : channelDesc.header.size) - sizeof(CommAbiHeader);
    CHK_SAFETY_FUNC_RET(memcpy_s(reinterpret_cast<uint8_t *>(&channelDescFinal) + sizeof(CommAbiHeader), copySize,
        reinterpret_cast<const uint8_t *>(&channelDesc) + sizeof(CommAbiHeader), copySize));
 
    if (channelDesc.header.version >= HCCL_CHANNEL_VERSION_ONE) {
        CHK_RET(ProcessHcclChannelDesc(channelDesc, channelDescFinal, hcclComm));
    }
 
    if (channelDesc.header.version > HCCL_CHANNEL_VERSION) {
        // 传入的版本高于当前版本，警告不支持的配置项将被忽略
        HCCL_WARNING("The version of provided [%u] is higher than the current version[%u], "
            "unsupported configuration will be ignored.",
            channelDesc.header.version, HCCL_CHANNEL_VERSION);
    } else if (channelDesc.header.version < HCCL_CHANNEL_VERSION) {
        // 传入的版本低于当前版本，警告高版本支持的配置项将被忽略
        HCCL_WARNING("The version of provided [%u] is lower than the current version[%u], "
            "configurations supported by later versions will be ignored.",
            channelDesc.header.version, HCCL_CHANNEL_VERSION);
    }
 
    // 如果扩展到version=2后
    // 1) 在底层为新的结构体和版本（version为2）上，会正常执行下面的判断处理逻辑；
    // 2) 在底层为旧的结构体和版本（version为1）上，下面的逻辑没有，version的2 > 1的部分会被忽略掉；
    if (channelDesc.header.version >= 2) {
    }
 
    return HCCL_SUCCESS;
}

static HcclResult BuildAivDeviceChannelEntity(const HcclChannelDesc &channelDesc, ChannelHandle hostChannel,
    ChannelHandle &deviceChannel)
{
    void *channel = nullptr;
    CHK_RET(hcomm::ChannelProcess::ChannelGet(hostChannel, &channel));
    hcomm::Channel *baseChannel = static_cast<hcomm::Channel *>(channel);
    CHK_PTR_NULL(baseChannel);

    if (channelDesc.channelProtocol == COMM_PROTOCOL_ROCE) {
        auto *aicpuTsRoceChannelV2 = dynamic_cast<hcomm::AicpuTsRoceChannelV2 *>(baseChannel);
        CHK_PTR_NULL(aicpuTsRoceChannelV2);
        HCCL_INFO("[%s] build AIV direct device channel by AICPU+Host RoCE flow, protocol[%d], "
            "hostHandle[0x%llx]", __func__, channelDesc.channelProtocol,
            static_cast<unsigned long long>(hostChannel));
        CHK_RET(aicpuTsRoceChannelV2->BuildAndGetDevChannelEntity(&deviceChannel));
        return HCCL_SUCCESS;
    }

    if (channelDesc.channelProtocol == COMM_PROTOCOL_UBC_CTP ||
        channelDesc.channelProtocol == COMM_PROTOCOL_UBC_TP) {
        auto *aivUrmaChannel = dynamic_cast<hcomm::AivUrmaChannel *>(baseChannel);
        CHK_PTR_NULL(aivUrmaChannel);
        HCCL_INFO("[%s] build AIV direct device channel by AIV+URMA flow, protocol[%d], "
            "hostHandle[0x%llx]", __func__, channelDesc.channelProtocol,
            static_cast<unsigned long long>(hostChannel));
        void *devChannelEntity = nullptr;
        CHK_RET(aivUrmaChannel->BuildChannelEntityToDevice(&devChannelEntity));
        CHK_PTR_NULL(devChannelEntity);
        deviceChannel = static_cast<ChannelHandle>(reinterpret_cast<uintptr_t>(devChannelEntity));
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[%s] protocol[%d] is not AIV direct channel protocol", __func__, channelDesc.channelProtocol);
    return HCCL_E_PARA;
}

static HcclResult ConvertAivChannelHandlesToDevicePtrs(CommEngine engine, const HcclChannelDesc *channelDescs,
    uint32_t channelNum, ChannelHandle *channels)
{
    if (engine != COMM_ENGINE_AIV) {
        return HCCL_SUCCESS;
    }

    std::vector<ChannelHandle> hostChannels(channels, channels + channelNum);
    std::vector<ChannelHandle> deviceChannels(hostChannels);
    std::vector<ChannelHandle> mappedDeviceChannels;
    std::vector<ChannelHandle> mappedHostChannels;
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        if (channelDescs[idx].channelProtocol != COMM_PROTOCOL_ROCE &&
            channelDescs[idx].channelProtocol != COMM_PROTOCOL_UBC_CTP &&
            channelDescs[idx].channelProtocol != COMM_PROTOCOL_UBC_TP) {
            continue;
        }
        CHK_RET(BuildAivDeviceChannelEntity(channelDescs[idx], hostChannels[idx], deviceChannels[idx]));
        mappedDeviceChannels.emplace_back(deviceChannels[idx]);
        mappedHostChannels.emplace_back(hostChannels[idx]);
        HCCL_INFO("[%s] convert AIV channel success, idx[%u], protocol[%d], hostHandle[0x%llx], devEntity[0x%llx]",
            __func__, idx, channelDescs[idx].channelProtocol, static_cast<unsigned long long>(hostChannels[idx]),
            static_cast<unsigned long long>(deviceChannels[idx]));
    }

    if (!mappedDeviceChannels.empty()) {
        CHK_RET(hcomm::ChannelProcess::RegisterChannelD2HMap(mappedDeviceChannels.data(), mappedHostChannels.data(),
            static_cast<uint32_t>(mappedDeviceChannels.size())));
    }

    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        channels[idx] = deviceChannels[idx];
    }
    return HCCL_SUCCESS;
}
static bool IsUbUrmaChannelProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UBOE;
}

static bool HasUbUrmaChannel(const std::vector<HcclChannelDesc> &channelDescFinals)
{
    for (const HcclChannelDesc &channelDesc : channelDescFinals) {
        if (IsUbUrmaChannelProtocol(channelDesc.channelProtocol)) {
            return true;
        }
    }
    return false;
}

static void AppendUniqueMemHandle(std::vector<HcclMemHandle> &mergedHandles, HcclMemHandle memHandle)
{
    if (memHandle == nullptr) {
        return;
    }
    if (std::find(mergedHandles.begin(), mergedHandles.end(), memHandle) == mergedHandles.end()) {
        mergedHandles.emplace_back(memHandle);
    }
}

static HcclResult MergeSymmetricMemHandles(HcclChannelDesc &channelDesc,
    const std::vector<HcclMemHandle> &symmetricMemHandles, std::vector<HcclMemHandle> &mergedHandles)
{
    if (!IsUbUrmaChannelProtocol(channelDesc.channelProtocol)) {
        return HCCL_SUCCESS;
    }
    if (channelDesc.memHandleNum != 0) {
        CHK_PTR_NULL(channelDesc.memHandles);
        for (uint32_t handleIdx = 0; handleIdx < channelDesc.memHandleNum; ++handleIdx) {
            AppendUniqueMemHandle(mergedHandles, channelDesc.memHandles[handleIdx]);
        }
    }
    for (HcclMemHandle memHandle : symmetricMemHandles) {
        AppendUniqueMemHandle(mergedHandles, memHandle);
    }
    CHK_PRT_RET(mergedHandles.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        HCCL_ERROR("[MergeSymmetricMemHandles] merged memHandleNum[%zu] exceeds uint32 max.",
            mergedHandles.size()), HCCL_E_PARA);
    channelDesc.memHandles = mergedHandles.data();
    channelDesc.memHandleNum = static_cast<uint32_t>(mergedHandles.size());
    return HCCL_SUCCESS;
}

static HcclResult AppendSymmetricMemHandles(hccl::CollComm *collComm,
    std::vector<HcclChannelDesc> &channelDescFinals,
    std::vector<std::vector<HcclMemHandle>> &mergedMemHandles,
    bool &hasSymmetricMemHandles)
{
    CHK_PTR_NULL(collComm);
    hasSymmetricMemHandles = false;
    if (!HasUbUrmaChannel(channelDescFinals)) {
        return HCCL_SUCCESS;
    }
    // 只有UB/URMA类channel需要追加symmetric memHandle参与建链交换。
    std::vector<HcclMemHandle> symmetricMemHandles;
    CHK_RET(collComm->RegisterPendingSymmetricMemHandles(symmetricMemHandles));
    if (symmetricMemHandles.empty()) {
        return HCCL_SUCCESS;
    }
    hasSymmetricMemHandles = true;

    mergedMemHandles.clear();
    mergedMemHandles.resize(channelDescFinals.size());
    for (size_t idx = 0; idx < channelDescFinals.size(); ++idx) {
        CHK_RET(MergeSymmetricMemHandles(channelDescFinals[idx], symmetricMemHandles, mergedMemHandles[idx]));
    }
    HCCL_INFO("[AppendSymmetricMemHandles] append symmetric memHandles success, channelNum[%zu], symMemHandleNum[%zu], "
        "protocols[UBC_CTP/UBC_TP/UBOE].",
        channelDescFinals.size(), symmetricMemHandles.size());
    return HCCL_SUCCESS;
}

static HcclResult UpdateSymmetricRemoteMems(hccl::CollComm *collComm, hccl::MyRank *myRank,
    const std::vector<HcclChannelDesc> &channelDescFinals, const ChannelHandle *channels, uint32_t channelNum)
{
    CHK_PTR_NULL(collComm);
    CHK_PTR_NULL(myRank);
    CHK_PTR_NULL(channels);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        const HcclChannelDesc &channelDesc = channelDescFinals[idx];
        if (!IsUbUrmaChannelProtocol(channelDesc.channelProtocol)) {
            continue;
        }
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        uint32_t memNum = 0;
        // CreateChannels完成后，从channel取回交换到的remoteMem/memTag并回填window。
        CHK_RET(myRank->ChannelGetRemoteMems(channels[idx], &memNum, &remoteMems, &memTags));
        if (memNum == 0) {
            continue;
        }
        CHK_RET(collComm->UpdateSymmetricRemoteMem(channelDesc.remoteRank, remoteMems, memTags, memNum));
    }
    return HCCL_SUCCESS;
}

bool CheckCommEngine(const CommEngine engine, const uint32_t opExpansionMode)
{
    constexpr uint32_t DEFAULT_MODE = 0;
    constexpr uint32_t CCU_MS_MODE = 5;
    constexpr uint32_t CCU_SCHE_MODE = 6;
    if (engine == CommEngine::COMM_ENGINE_CCU) {
        return opExpansionMode == DEFAULT_MODE
            || opExpansionMode == CCU_MS_MODE
            || opExpansionMode == CCU_SCHE_MODE;
    }

    return true;
}

constexpr uint32_t CHANNEL_NUM_MAX = 1024 * 1024;  // channel的默认限制最大为1024 * 1024

HcclResult RegisterToClusterMonitor(HcclComm comm)
{
    HCCL_INFO("[%s] START, comm[%p].", __func__, comm);
    CHK_PRT_RET(comm == nullptr,  HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("comm is not support [%s]", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    CHK_RET(CollCommMgr::GetInstance()->GetClusterMonitor(collComm->GetDeviceLogicId()).RegisterToClusterMonitor(comm));
    HCCL_RUN_INFO("%s Success", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, 
    const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    HcclUs startut = TIME_NOW();
    u64 beginTime =  Hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    EXCEPTION_HANDLE_BEGIN

    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0 || channelNum > CHANNEL_NUM_MAX), 
        HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u], max channel num[%u]",
        __func__, channelNum, CHANNEL_NUM_MAX), HCCL_E_PARA
    );
 
    HcclResult ret = HCCL_SUCCESS;
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    HCCL_RUN_INFO("Entry-%s channelNum[%u], engine[%s] group[%s]", __func__, channelNum, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), hcclComm->GetIdentifier().c_str());
    std::vector<HcclChannelDesc> channelDescFinals;
    std::vector<std::vector<HcclMemHandle>> mergedMemHandles;
    for (uint32_t idx = 0; idx < channelNum; idx++) {
        HcclChannelDesc channelDescFinal;
        HcclChannelDescInit(&channelDescFinal, 1);
        ret = ProcessHcclResPackReq(channelDescs[idx], channelDescFinal, hcclComm);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("ProcessHcclResPackReq failed. channelDesc idx[%u], group[%s], engine[%s] channelNum[%llu], ret[%d]", idx, hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, ret), ret);
        channelDescFinals.push_back(channelDescFinal);
    }
 
    if (hcclComm->IsCommunicatorV2()) {  // A5
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        const std::string &commTag = hcclComm->GetIdentifier();
        hccl::MyRank* myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
 
        const uint32_t opExpansionMode = myRank->GetOpExpansionMode();
        if (!CheckCommEngine(engine, opExpansionMode)) {
            HCCL_ERROR("[%s] failed, coll comm[%p] group[%s] opExpansionMode[%d] is not supported by CCU engine[%s].", 
                __func__, hcclComm, hcclComm->GetIdentifier().c_str(), opExpansionMode, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (engine != CommEngine::COMM_ENGINE_CPU) { // host dpu场景暂不支持cluster monitor
            ret = RegisterToClusterMonitor(comm);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("RegisterToClusterMonitor failed. group[%s], engine[%s], channelNum[%llu], ret[%d]", hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, ret), ret);
        }

        if (!GetDebugConfigInited()) {
            InitDebugConfigByEnv();
        }

        bool hasSymmetricMemHandles = false;
        CHK_RET(AppendSymmetricMemHandles(collComm, channelDescFinals, mergedMemHandles, hasSymmetricMemHandles));
        HCCL_INFO("[HcclChannelAcquire] AppendSymmetricMemHandles done, group[%s], engine[%d], channelNum[%u], "
            "hasSymmetricMemHandles[%d], mergedMemHandleGroups[%zu].",
            commTag.c_str(), engine, channelNum, hasSymmetricMemHandles, mergedMemHandles.size());
        ret = myRank->CreateChannels(engine, commTag, channelDescFinals.data(), channelNum, channels);
        CHK_PRT_RET((ret == HCCL_E_AGAIN || ret == HCCL_E_UNAVAIL),
            HCCL_WARNING("CreateChannels group[%s], engine[%s] ret[%d]", commTag.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret), ret);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("CreateChannels failed. group[%s], engine[%s] ret[%d]", commTag.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret), ret);
        if (hasSymmetricMemHandles) {
            CHK_RET(UpdateSymmetricRemoteMems(collComm, myRank, channelDescFinals, channels, channelNum));
        }
        if (engine == COMM_ENGINE_CPU) {
            HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
            CHK_PTR_NULL(hcclCommDfx);
            auto callback = hcclCommDfx->GetDpuCallback();
            for (uint32_t idx = 0; idx < channelNum; idx++) {
                int32_t ret = HcommDpuChannelRegisterDfx(channels[idx], callback);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[HcclChannelAcquire] group[%s] Failed to register DFX callback for channel[%u], ret[%d]", commTag.c_str(), idx, ret),
                    static_cast<HcclResult>(ret));
            }
            HCCL_INFO("[HcclChannelAcquire] group[%s] channelNum[%u] Register DFX callback for CPU channels success", commTag.c_str(), channelNum);
        }
        if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
            HCCL_INFO("[HcclChannelAcquire] ReportChannelAicpuKernel start");
            HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
            CHK_PTR_NULL(hcclCommDfx);
            std::string kernelName = "RunAicpuIndOpChannelInitV2";
            // 还是kernel的当前无法判断
            ret = hcclCommDfx->ReportKernel(beginTime, commTag, kernelName, SalGetTid(), false);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[HcclChannelAcquire] group[%s] Failed to report kernel for kernelName[%s], tid[%d], ret[%d]", commTag.c_str(), kernelName.c_str(), SalGetTid(), ret), ret);
        }
    } else {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        if (collComm != nullptr) {
            hccl::MyRank *myRank = collComm->GetMyRank();
            if (hcclComm->GetConnectMode() && engine == COMM_ENGINE_CPU && myRank != nullptr) {
                const std::string &commTag = hcclComm->GetIdentifier();
                ret = myRank->CreateChannels(engine, commTag, channelDescFinals.data(), channelNum, channels);
            } else {
                auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
                ret = channelMgr.ChannelCommCreate(hcclComm->GetIdentifier(), engine,
                    channelDescFinals.data(), channelNum, channels);
            }
        } else {
            auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
            ret = channelMgr.ChannelCommCreate(hcclComm->GetIdentifier(), engine,
                channelDescFinals.data(), channelNum, channels);
        }
    }
 
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] Failed to acquire channel, group[%s], engine[%s], channelNum[%llu], ret[%d]", __func__, hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, ret), ret);

    CHK_RET(ConvertAivChannelHandlesToDevicePtrs(engine, channelDescFinals.data(), channelNum, channels));
 
    HCCL_RUN_INFO("[%s] acquire channel success, group[%s], engine[%s], channelNum[%llu], take time [%lld]us.", __func__, hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, DURATION_US(TIME_NOW() - startut));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclGroupStart()
{
    return HcclLegacyGroupStart();
}

HcclResult HcclGroupEndV2()
{
    CHK_RET(groupLaunchA5());
    HCCL_INFO("[GroupEnd] to the end");
    return HCCL_SUCCESS;
}

HcclResult HcclGroupEnd()
{
    if (hcclGroupDepth == 0) {
        HCCL_ERROR("HcclGroupEnd: not in a group call. Didn't call HcclGroupStart before.");
        return HCCL_E_NOT_SUPPORT;
    }
    if (--hcclGroupDepth > 0) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[HcclGroupEnd] hcclGroupDepth=[%d]", hcclGroupDepth);
    /*遇到最后一个HcclGroupEnd才处理group内的所有任务*/
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        return HcclGroupEndV2();
    }());
    return HcclLegacyGroupEnd();
}

HcclResult HcclGroupStatusGet(bool *isGroupEnabled)
{
    CHK_PTR_NULL(isGroupEnabled);
    *isGroupEnabled = (hcclGroupDepth > 0);
    return HCCL_SUCCESS;
}
