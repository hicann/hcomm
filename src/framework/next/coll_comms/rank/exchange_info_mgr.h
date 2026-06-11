/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXCHANGE_INFO_MGR_H
#define EXCHANGE_INFO_MGR_H

#include "hcomm_res_defs.h"
#include "common.h"
#include "coll_comm_config_consistency.h"
#include "socket/socket.h"

namespace hccl {
class ExchangeInfoMgr {
public:
    ExchangeInfoMgr();
    ~ExchangeInfoMgr();

    HcclResult BatchExchangeAndCheckConsistency(
        const HcclChannelDesc* channelDescs,
        const std::vector<HcommChannelDesc> &hcommDescs,
        uint32_t channelNum,
        CollCommConfigConsistency &collCommConfigConsistency,
        const std::string &commTag);
    HcclResult ExchangeUserInfo(
        const std::vector<Hccl::Socket*> &sockets,
        const std::vector<u32> &remoteRanks,
        const std::vector<HcommSocketRole> &roles,
        CollCommConfigConsistency &collCommConfigConsistency);
    HcclResult BatchExchangeFixedData(
        const std::vector<Hccl::Socket*> &sockets,
        const std::vector<u32> &remoteRanks,
        const std::vector<HcommSocketRole> &roles,
        const u8 *sendData, u32 sendLen,
        u8 *recvData, u32 recvLen);
    HcclResult WaitAllAsyncComplete(const std::vector<Hccl::Socket*> &sockets,
        const std::vector<u32> &remoteRanks) const;
    HcclResult WaitActiveAsyncComplete(
        const std::vector<Hccl::Socket*> &sockets,
        const std::vector<u32> &remoteRanks,
        const std::vector<HcommSocketRole> &roles,
        const std::vector<u32> &remoteExchangeInfoLens,
        u32 localExchangeInfoLen,
        bool isFirstPass);
};
} // namespace hccl

#endif // EXCHANGE_INFO_MGR_H