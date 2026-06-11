/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "base_config.h"
#include <fstream>
#include "sal.h"
#include "log.h"
#include "orion_adapter_rts.h"

namespace Hccl {

// EnvHostNicConfig

void EnvHostNicConfig::Parse()
{
    hcclIfIp.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_IF_IP set by %s to [%s]", hcclIfIp.GetSource(),
        GetControlIfIp().Describe().c_str());

    hcclIfBasePort.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_IF_BASE_PORT set by %s to [%u]", hcclIfBasePort.GetSource(),
        GetIfBasePort());

    hcclSocketIfName.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_SOCKET_IFNAME set by %s to [%s]", hcclSocketIfName.GetSource(),
        GetSocketIfName().configIfNameStr.c_str());

    whitelistDisable.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_WHITELIST_DISABLE set by %s to [%d]", whitelistDisable.GetSource(),
        whitelistDisable.Get());

    if (!whitelistDisable.Get()) {
        hcclWhiteListFile.Parse();
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_WHITELIST_FILE set by %s to [%s]", hcclWhiteListFile.GetSource(),
            GetWhiteListFile().c_str());
    }

    hcclHostSocketPortRange.Parse();
    std::ostringstream hosrPortRangeOss;
    for (auto range : GetHostSocketPortRange()) {
        hosrPortRangeOss << "[" << range.min << ", " << range.max << "]";
    }
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_HOST_SOCKET_PORT_RANGE set by %s to %s", hcclHostSocketPortRange.GetSource(),
        hosrPortRangeOss.str().c_str());

    hcclDeviceSocketPortRange.Parse();
    std::ostringstream devicePortRangeOss;
    for (auto range : GetDeviceSocketPortRange()) {
        devicePortRangeOss << "[" << range.min << ", " << range.max << "]";
    }
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_NPU_SOCKET_PORT_RANGE set by %s to %s", hcclDeviceSocketPortRange.GetSource(),
        devicePortRangeOss.str().c_str());
}

const IpAddress &EnvHostNicConfig::GetControlIfIp() const
{
    return hcclIfIp.Get();
}

u32 EnvHostNicConfig::GetIfBasePort() const
{
    return hcclIfBasePort.Get();
}

const SocketIfName &EnvHostNicConfig::GetSocketIfName() const
{
    return hcclSocketIfName.Get();
}

bool EnvHostNicConfig::GetWhitelistDisable() const
{
    return whitelistDisable.Get();
}

const std::string &EnvHostNicConfig::GetWhiteListFile() const
{
    return hcclWhiteListFile.Get();
}

const std::vector<SocketPortRange> &EnvHostNicConfig::GetHostSocketPortRange() const
{
    return hcclHostSocketPortRange.Get();
}

const std::vector<SocketPortRange> &EnvHostNicConfig::GetDeviceSocketPortRange() const
{
    return hcclDeviceSocketPortRange.Get();
}

// EnvSocketConfig

void EnvSocketConfig::Parse()
{
    hcclSocketFamily.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_SOCKET_FAMILY set by %s to [%d]", hcclSocketFamily.GetSource(),
        GetSocketFamily());

    linkTimeOut.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_CONNECT_TIMEOUT set by %s to [%d]s", linkTimeOut.GetSource(),
        GetLinkTimeOut());
}

s32 EnvSocketConfig::GetSocketFamily() const
{
    return hcclSocketFamily.Get();
}

s32 EnvSocketConfig::GetLinkTimeOut() const
{
    return linkTimeOut.Get();
}

// EnvRtsConfig

void EnvRtsConfig::Parse()
{
    execTimeOut.Parse();
    aivExecTimeOut.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_EXEC_TIMEOUT set by %s to [%u]s", execTimeOut.GetSource(),
    GetExecTimeOut());
}

u32 EnvRtsConfig::GetExecTimeOut() const
{
    return execTimeOut.Get();
}

double EnvRtsConfig::GetAivExecTimeOut() const
{
    return aivExecTimeOut.Get();
}

// EnvRdmaConfig

void EnvRdmaConfig::Parse()
{
    rdmaTrafficClass.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TC set by %s to [%u]", rdmaTrafficClass.GetSource(),
        GetRdmaTrafficClass());

    rdmaServerLevel.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_SL set by %s to [%u]", rdmaServerLevel.GetSource(),
        GetRdmaServerLevel());

    rdmaTimeOut.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TIMEOUT set by %s to [%u]", rdmaTimeOut.GetSource(),
        GetRdmaTimeOut());

    rdmaRetryCnt.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_RETRY_CNT set by %s to [%u]", rdmaRetryCnt.GetSource(),
        GetRdmaRetryCnt());

    uboeTimeOut.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_UBOE_TIMEOUT set by %s to [%u]", uboeTimeOut.GetSource(),
        GetUboeTimeOut());

    ubTimeOut.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_UB_TIMEOUT set by %s to [%u]", ubTimeOut.GetSource(),
        GetUbTimeOut());

    queueNum.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_QPS_PER_CONNECTION set by %s to [%u]", queueNum.GetSource(),
        GetRdmaQueueNum());

    multiQpThreshold.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_MULTI_QP_THRESHOLD set by %s to [%u]B", multiQpThreshold.GetSource(),
        GetRdmaMultiQpThreshold());
}

u32 EnvRdmaConfig::GetRdmaTrafficClass() const
{
    return rdmaTrafficClass.Get();
}

u32 EnvRdmaConfig::GetRdmaServerLevel() const
{
    return rdmaServerLevel.Get();
}

u32 EnvRdmaConfig::GetRdmaTimeOut() const
{
    return rdmaTimeOut.Get();
}

u32 EnvRdmaConfig::GetRdmaRetryCnt() const
{
    return rdmaRetryCnt.Get();
}

u32 EnvRdmaConfig::GetRdmaQueueNum() const
{
    return queueNum.Get();
}

u32 EnvRdmaConfig::GetRdmaMultiQpThreshold() const
{
    return multiQpThreshold.Get();
}

u32 EnvRdmaConfig::GetUboeTimeOut() const
{
    return uboeTimeOut.Get();
}

u32 EnvRdmaConfig::GetUbTimeOut() const
{
    return ubTimeOut.Get();
}

// EnvAlgoConfig

void EnvAlgoConfig::Parse()
{
    primQueueGenName.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] PRIM_QUEUE_GEN_NAME set by %s to [%s]", primQueueGenName.GetSource(),
        GetPrimQueueGenName().c_str());

    hcclAlgoConfig.Parse();
    std::ostringstream hcclAlgoConfigOss;
    for (auto algoConfig : GetAlgoConfig()) {
        OpType opType = algoConfig.first;
        hcclAlgoConfigOss << "[" << opType.Describe().c_str() << ", ";
        std::vector<HcclAlgoType> algoTypes = algoConfig.second;
        for (auto algoType : algoTypes) {
            hcclAlgoConfigOss << algoType.Describe().c_str() << " ";
        }
        hcclAlgoConfigOss << "]";
    }
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_ALGO set by %s to [%s]", hcclAlgoConfig.GetSource(),
        hcclAlgoConfigOss.str().c_str());

    bufferSize.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_BUFFSIZE set by %s to [%llu]B", bufferSize.GetSource(),
        GetBuffSize());

    hcclAccelerator_.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_OP_EXPANSION_MODE set by %s to [%s]", hcclAccelerator_.GetSource(),
        GetHcclAccelerator().Describe().c_str());
}

const std::string &EnvAlgoConfig::GetPrimQueueGenName() const
{
    return primQueueGenName.Get();
}

const std::map<OpType, std::vector<HcclAlgoType>> &EnvAlgoConfig::GetAlgoConfig() const
{
    return hcclAlgoConfig.Get();
}

u64 EnvAlgoConfig::GetBuffSize() const
{
    return bufferSize.Get();
}

HcclAccelerator EnvAlgoConfig::GetHcclAccelerator() const
{
    return hcclAccelerator_.Get();
}
 
// EnvLogConfig
void EnvLogConfig::Parse()
{
    entryLogEnable.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_ENTRY_LOG_ENABLE set by %s to [%d]", entryLogEnable.GetSource(),
        GetEntryLogEnable());

    cannVersion.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] LD_LIBRARY_PATH set by %s to [%s]", cannVersion.GetSource(),
        GetCannVersion().c_str());

    dfsConfig.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_DFS_CONFIG task_exception set by %s to [%d], cluster_heartbeat set by %s to [%d], "
                  "rankConsistentState set by %s to [%d]",
        dfsConfig.GetSource(), GetDfsConfig().taskExceptionEnable, dfsConfig.GetSource(),
        GetDfsConfig().clusterHeartBeatEnable, dfsConfig.GetSource(), GetDfsConfig().rankConsistentState);
}

bool EnvLogConfig::GetEntryLogEnable() const
{
    return entryLogEnable.Get();
}

const std::string &EnvLogConfig::GetCannVersion() const
{
    return cannVersion.Get();
}

const DfsConfig &EnvLogConfig::GetDfsConfig() const
{
    return dfsConfig.Get();
}

// EnvDetourConfig

void EnvDetourConfig::Parse()
{
    detourType.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_DETOUR set by %s to [%s]", detourType.GetSource(),
        GetDetourType().Describe().c_str());
}

HcclDetourType EnvDetourConfig::GetDetourType() const
{
    return detourType.Get();
}

} // namespace Hccl
