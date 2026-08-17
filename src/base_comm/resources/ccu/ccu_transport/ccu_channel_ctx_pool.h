/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_CHANNELCTX_POOLS_H
#define CCU_CHANNELCTX_POOLS_H

#include <vector>
#include <unordered_map>

#include "ccu_jetty_.h"
#include "hash_utils.h"
#include "ip_address.h"
#include "virtual_topo.h"

namespace hcomm {

// 管理着有限的硬件资源：ChannelCtx与jetty
class CcuChannelCtxPool final {
public:
    explicit CcuChannelCtxPool(int32_t devLogicId);
    ~CcuChannelCtxPool();

    HcclResult PrepareCreate(const std::vector<Hccl::LinkData>& links, uint32_t sqSize = 0);
    using CcuChannelCtx = std::pair<CcuChannelInfo, std::vector<CcuJetty*>>;
    HcclResult GetChannelCtx(const Hccl::LinkData& link, CcuChannelCtx& channelCtx) const;
    HcclResult GetCcuChannelCtxById(const std::pair<uint8_t, uint32_t>& key, CcuChannelCtx& ctx);

private:
    struct ResIdHash {
        std::size_t operator()(const std::pair<uint8_t, uint32_t>& p) const
        {
            return Hccl::HashCombine({p.first, p.second});
        }
    };

    using CcuJettyPtr = CcuJetty*;
    using BatchKey = Hccl::IpAddress; // srcIpAddress;
    using ResIdkey = std::pair<uint8_t, uint32_t>;
    using ChannelIdKey = ResIdkey;
    using JettyIdKey = ResIdkey;

    // 平台层每次调用CcuAllocChannels可能提供多个ccu channel，且不同srcIp的jetty不能复用
    // 故以srcIp为粒度，多次调用接口，每次接口结果定义为一个批次资源
    struct ResourceBatch { // 记录该批次申请到的所有channel资源信息
        BatchKey key;
        std::vector<ChannelIdKey> channelIdKeys;
        std::vector<ChannelIdKey> availableChannelIdKeys;
        std::unordered_map<JettyIdKey, std::unique_ptr<CcuJetty>, ResIdHash> jettys;

        ResourceBatch(const BatchKey& batchKey) : key(batchKey) {};
        HcclResult Init(const std::vector<CcuChannelInfo>& channelInfos);
    };

    struct Allocation {
        Hccl::LinkData link;
        ChannelIdKey channelIdKey;
        ResourceBatch* batchPtr;
    };

    struct UnconfirmedRecord {
        std::vector<Allocation> allocations;            // 记录从已申请的资源中的分配操作
        std::unordered_set<ResourceBatch*> newBatchSet; // 记录新申请资源的操作

        void Clear()
        {
            allocations.clear();
            newBatchSet.clear();
        }
    };

private:
    HcclResult GetAvailableBatch(const BatchKey& batchKey, ResourceBatch*& batchPtr, uint32_t sqSize);
    bool FindAvailableBatch(const BatchKey& batchKey, ResourceBatch*& batchPtr) const;
    HcclResult CreateAndSaveNewBatch(
        const BatchKey& batchKey, const std::vector<CcuChannelInfo> channelInfos, ResourceBatch*& batchPtr);
    HcclResult ReleaseConfirmedChannelRes();

private:
    int32_t devLogicId_{0};
    bool isReleased_{true};

    // 本轮下发算子新增分配记录
    UnconfirmedRecord unconfirmedRecord_;
    // 各资源申请记录，当前按SrcIpAddr粒度申请和管理
    std::unordered_map<BatchKey, std::vector<std::unique_ptr<ResourceBatch>>> batchMap_;
    // 各link已分配的channel资源Id信息
    std::unordered_map<Hccl::LinkData, ChannelIdKey> allocatedChannelIdMap_;
    // 全部已申请的channel资源信息，资源申请成功后将要记录到该map中
    std::unordered_map<ChannelIdKey, CcuChannelCtx, ResIdHash> channelJettyInfoMap_;
    // 以die粒度记录已分配channel资源, index: dieId
    std::unordered_map<uint8_t, uint32_t> usedChannelCntMap_;
    // 记录channel与对端rank的映射关系, index: (die, channelId)
    std::unordered_map<ChannelIdKey, Hccl::RankId, ResIdHash> channelRemoteRankIdMap_;
};

} // namespace hcomm

#endif // CCU_CHANNELCTX_POOLS_H
