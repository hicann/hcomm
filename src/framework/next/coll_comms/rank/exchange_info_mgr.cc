/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exchange_info_mgr.h"
#include "env_config/env_config.h"

namespace hccl {
ExchangeInfoMgr::ExchangeInfoMgr()
{
}

ExchangeInfoMgr::~ExchangeInfoMgr()
{
    HCCL_INFO("[ExchangeInfoMgr][~ExchangeInfoMgr] CollCommConfigConsistency deinit");
}

HcclResult ExchangeInfoMgr::BatchExchangeAndCheckConsistency(
    const HcclChannelDesc* channelDescs,
    const std::vector<HcommChannelDesc> &hcommDescs,
    uint32_t channelNum,
    CollCommConfigConsistency &collCommConfigConsistency,
    const std::string &commTag)
{
    std::vector<Hccl::Socket*> sockets;
    std::vector<u32> remoteRanks;
    std::vector<HcommSocketRole> roles;

    if (channelNum == 0) {
        HCCL_INFO("[BatchExchangeAndCheckConsistency] channelNum is 0.");
        return HCCL_SUCCESS;
    }

    for (uint32_t i = 0; i < channelNum; i++) {
        u32 remoteRank = channelDescs[i].remoteRank;
        HcommSocket rawSocket = hcommDescs[i].socket;
        Hccl::Socket *socket = static_cast<Hccl::Socket *>(rawSocket);
        CHK_PRT_RET(socket == nullptr,
            HCCL_ERROR("[BatchExchangeAndCheckConsistency] socket is null for channel[%u] remoteRank[%u].",
                i, remoteRank),
            HCCL_E_INTERNAL);
        sockets.push_back(socket);
        remoteRanks.push_back(remoteRank);
        roles.push_back(hcommDescs[i].role);
    }

    // 交换HCCL算子信息 ======
    CHK_RET(ExchangeUserInfo(sockets, remoteRanks, roles, collCommConfigConsistency));
    CHK_RET(collCommConfigConsistency.ResetExchangeInfo());

    return HCCL_SUCCESS;
}

HcclResult ExchangeInfoMgr::ExchangeUserInfo(
    const std::vector<Hccl::Socket*> &sockets,
    const std::vector<u32> &remoteRanks,
    const std::vector<HcommSocketRole> &roles,
    CollCommConfigConsistency &collCommConfigConsistency)
{
    u32 localExchangeInfoLen = collCommConfigConsistency.GetExchangeInfoLen();
    if (localExchangeInfoLen == 0) {
        HCCL_INFO("[ExchangeUserInfo] localExchangeInfoLen is 0.");
        return HCCL_SUCCESS;
    }
    // 交换infoLen
    std::vector<u32> remoteExchangeInfoLens(sockets.size(), 0);
    CHK_RET(BatchExchangeFixedData(sockets, remoteRanks, roles,
        reinterpret_cast<const u8*>(&localExchangeInfoLen), sizeof(u32),
        reinterpret_cast<u8*>(remoteExchangeInfoLens.data()), sizeof(u32)));

    // 交换info数据（长度可能不同，需逐个收发）
    std::vector<std::vector<u8>> remoteUserDatas(sockets.size());
    // SERVER先Recv/CLIENT先Send
    for (u32 i = 0; i < sockets.size(); i++) {
        if (roles[i] == HCOMM_SOCKET_ROLE_SERVER) {
            remoteUserDatas[i].resize(remoteExchangeInfoLens[i], 0);
            sockets[i]->RecvAsync(remoteUserDatas[i].data(), remoteExchangeInfoLens[i]);
        } else {
            std::vector<u8> exchangeBuf;
            collCommConfigConsistency.GetExchangeInfoBuf(exchangeBuf);
            sockets[i]->SendAsync(exchangeBuf.data(), localExchangeInfoLen);
        }
    }
    CHK_RET(WaitActiveAsyncComplete(sockets, remoteRanks, roles,
        remoteExchangeInfoLens, localExchangeInfoLen, true));

    // SERVER再Send/CLIENT再Recv
    for (u32 i = 0; i < sockets.size(); i++) {
        if (roles[i] == HCOMM_SOCKET_ROLE_SERVER) {
            std::vector<u8> exchangeBuf;
            collCommConfigConsistency.GetExchangeInfoBuf(exchangeBuf);
            sockets[i]->SendAsync(exchangeBuf.data(), localExchangeInfoLen);
        } else {
            remoteUserDatas[i].resize(remoteExchangeInfoLens[i], 0);
            sockets[i]->RecvAsync(remoteUserDatas[i].data(), remoteExchangeInfoLens[i]);
        }
    }
    CHK_RET(WaitActiveAsyncComplete(sockets, remoteRanks, roles,
        remoteExchangeInfoLens, localExchangeInfoLen, false));

    // 存储对端交换信息
    for (u32 i = 0; i < sockets.size(); i++) {
        if (remoteExchangeInfoLens[i] > 0 && !remoteUserDatas[i].empty()) {
            CHK_RET(collCommConfigConsistency.StoreRemoteExchangeInfo(remoteRanks[i], remoteUserDatas[i]));
        }
    }

    HCCL_INFO("[ExchangeUserInfo] suc.");
    return HCCL_SUCCESS;
}

// 批量异步交换定长数据（SERVER先Recv再Send，CLIENT先Send再Recv，防死锁）
HcclResult ExchangeInfoMgr::BatchExchangeFixedData(
    const std::vector<Hccl::Socket*> &sockets,
    const std::vector<u32> &remoteRanks,
    const std::vector<HcommSocketRole> &roles,
    const u8 *sendData, u32 sendLen,
    u8 *recvData, u32 recvLen)
{
    // SERVER先Recv/CLIENT先Send
    for (u32 i = 0; i < sockets.size(); i++) {
        if (roles[i] == HCOMM_SOCKET_ROLE_SERVER) {
            sockets[i]->RecvAsync(recvData + i * recvLen, recvLen);
        } else {
            sockets[i]->SendAsync(sendData, sendLen);
        }
    }
    CHK_RET(WaitAllAsyncComplete(sockets, remoteRanks));

    // SERVER再Send/CLIENT再Recv
    for (u32 i = 0; i < sockets.size(); i++) {
        if (roles[i] == HCOMM_SOCKET_ROLE_SERVER) {
            sockets[i]->SendAsync(sendData, sendLen);
        } else {
            sockets[i]->RecvAsync(recvData + i * recvLen, recvLen);
        }
    }
    CHK_RET(WaitAllAsyncComplete(sockets, remoteRanks));

    return HCCL_SUCCESS;
}

HcclResult ExchangeInfoMgr::WaitAllAsyncComplete(
    const std::vector<Hccl::Socket*> &sockets,
    const std::vector<u32> &remoteRanks) const
{
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    std::vector<bool> done(sockets.size(), false);
    uint32_t doneCount = 0;

    while (doneCount < sockets.size()) {
        for (size_t i = 0; i < sockets.size(); i++) {
            if (done[i]) {
                continue;
            }
            Hccl::SocketStatus status = sockets[i]->GetAsyncStatus();
            if (status == Hccl::SocketStatus::OK) {
                done[i] = true;
                doneCount++;
                continue;
            }
            if (status == Hccl::SocketStatus::TIMEOUT) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - startTime).count();
                HCCL_ERROR("[WaitAllAsyncComplete] socket timeout for remoteRank[%u], elapsed[%lld]ms.",
                    remoteRanks[i], elapsed);
                return HCCL_E_TIMEOUT;
            }
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            for (size_t i = 0; i < sockets.size(); i++) {
                if (!done[i]) {
                    HCCL_ERROR("[WaitAllAsyncComplete] wall-clock timeout for remoteRank[%u], elapsed[%lld]ms.",
                        remoteRanks[i], elapsed);
                }
            }
            return HCCL_E_TIMEOUT;
        }
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();
    HCCL_INFO("[WaitAllAsyncComplete] all[%zu] sockets completed, elapsed[%lld]ms.",
        sockets.size(), elapsed);
    return HCCL_SUCCESS;
}

// 收集并等待有实际异步操作的socket子集
HcclResult ExchangeInfoMgr::WaitActiveAsyncComplete(
    const std::vector<Hccl::Socket*> &sockets,
    const std::vector<u32> &remoteRanks,
    const std::vector<HcommSocketRole> &roles,
    const std::vector<u32> &remoteExchangeInfoLens,
    u32 localExchangeInfoLen,
    bool isFirstPass)
{
    std::vector<Hccl::Socket*> activeSockets;
    std::vector<u32> activeRanks;
    for (u32 i = 0; i < sockets.size(); i++) {
        bool isActive = isFirstPass
            ? (roles[i] == HCOMM_SOCKET_ROLE_SERVER && remoteExchangeInfoLens[i] > 0) ||
              (roles[i] != HCOMM_SOCKET_ROLE_SERVER && localExchangeInfoLen > 0)
            : (roles[i] == HCOMM_SOCKET_ROLE_SERVER && localExchangeInfoLen > 0) ||
              (roles[i] != HCOMM_SOCKET_ROLE_SERVER && remoteExchangeInfoLens[i] > 0);
        if (isActive) {
            activeSockets.push_back(sockets[i]);
            activeRanks.push_back(remoteRanks[i]);
        }
    }
    if (!activeSockets.empty()) {
        CHK_RET(WaitAllAsyncComplete(activeSockets, activeRanks));
    }
    return HCCL_SUCCESS;
}
}
