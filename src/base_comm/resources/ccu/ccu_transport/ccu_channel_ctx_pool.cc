/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_channel_ctx_pool.h"

#include <unordered_set>

#include "ccu_device_pub.h"
#include "orion_adpt_utils.h"

namespace hcomm {

constexpr uint32_t CCU_DEFAULT_REQUEST_SQ_SIZE = 128;
constexpr uint32_t CCU_DEFAULT_REQUEST_CHANNEL_NUM = 1;
constexpr uint32_t CCU_DEFAULT_REQUEST_JETTY_NUM = 0; // 申请数量为0时，由平台层决定提供数量

CcuChannelCtxPool::CcuChannelCtxPool(int32_t devLogicId) : devLogicId_(devLogicId) {}

CcuChannelCtxPool::~CcuChannelCtxPool()
{
    // 对象析构时清空多个map，batchMap_中元素的jettys清空触发ccuJetty析构释放
    (void)ReleaseConfirmedChannelRes();
}

HcclResult CcuChannelCtxPool::ResourceBatch::Init(const std::vector<CcuChannelInfo>& channelInfos)
{
    const uint32_t channelNum = channelInfos.size();
    channelIdKeys.reserve(channelNum);
    availableChannelIdKeys.reserve(channelNum);
    for (const auto& channelInfo : channelInfos) {
        const auto dieId = channelInfo.dieId;
        const auto channelId = channelInfo.channelId;
        channelIdKeys.emplace_back(dieId, channelId);
        availableChannelIdKeys.emplace_back(dieId, channelId);

        for (const auto& jettyInfo : channelInfo.jettyInfos) {
            const auto taJettyId = jettyInfo.taJettyId;
            const auto jettyIdKey = std::make_pair(dieId, taJettyId);
            if (jettys.find(jettyIdKey) != jettys.end()) {
                continue;
            }

            std::unique_ptr<CcuJetty> ccuJetty;
            CHK_RET(CcuCreateJetty(key, jettyInfo, ccuJetty));

            jettys[jettyIdKey] = std::move(ccuJetty);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::PrepareCreate(const std::vector<Hccl::LinkData>& links, uint32_t sqSize)
{
    CHK_PRT_RET(
        links.empty(),
        HCCL_INFO("[CcuChannelCtxPool][%s] passed, links is empty, devLogicId[%d].", __func__, devLogicId_),
        HcclResult::HCCL_SUCCESS);

    for (const auto& link : links) {
        auto it = allocatedChannelIdMap_.find(link);
        if (it != allocatedChannelIdMap_.end()) {
            HCCL_INFO(
                "[CcuChannelCtxPool][%s] passed, link[%s] is already allocated, "
                "devLogicId[%d].",
                __func__, link.Describe().c_str(), devLogicId_);
            continue;
        }

        const auto& locAddr = link.GetLocalAddr();
        ResourceBatch* batchPtr = nullptr;
        auto ret = GetAvailableBatch(locAddr, batchPtr, sqSize);
        CHK_PRT_RET(
            ret == HcclResult::HCCL_E_UNAVAIL,
            HCCL_WARNING(
                "[CcuChannelCtxPool][%s] failed to alloc ccu channels, ccu resources "
                "are unavailable, locAddr[%s], devLogicId[%d], sqSize[%u].",
                __func__, locAddr.Describe().c_str(), devLogicId_, sqSize),
            ret);
        CHK_RET(ret);

        ChannelIdKey channelIdKey = batchPtr->availableChannelIdKeys.back();
        batchPtr->availableChannelIdKeys.pop_back();
        unconfirmedRecord_.allocations.emplace_back(Allocation{link, channelIdKey, batchPtr});
        allocatedChannelIdMap_[link] = channelIdKey;
        channelRemoteRankIdMap_[channelIdKey] = link.GetRemoteRankId();

        HCCL_INFO(
            "[CcuChannelCtxPool][%s] allocated new channelId[%u] of die[%u] to link[%s], "
            "devLogicId[%d], sqSize[%u].",
            __func__, channelIdKey.second, channelIdKey.first, link.Describe().c_str(), devLogicId_, sqSize);
    }

    isReleased_ = false;
    return HcclResult::HCCL_SUCCESS;
}

// 当前以locAddr为粒度调用，根据locAddr可以找到已申请的批次，如果资源充足则复用，不足则按新批次申请资源
HcclResult CcuChannelCtxPool::GetAvailableBatch(const BatchKey& batchKey, ResourceBatch*& batchPtr, uint32_t sqSize)
{
    // 当前以locAddr作为batchKey，不同本端不能复用资源
    if (FindAvailableBatch(batchKey, batchPtr)) {
        return HcclResult::HCCL_SUCCESS;
    }
    // 已有的资源不足，需要新增资源，获取的channel数量可能超过申请数量
    CommAddr commAddr{};
    CHK_RET(IpAddressToCommAddr(batchKey, commAddr));
    // 使用传入的sqSize，如果为0xFFFFFFFF则使用默认值
    uint32_t actualSqSize = (sqSize != 0xFFFFFFFF) ? sqSize : CCU_DEFAULT_REQUEST_SQ_SIZE;
    const CcuChannelPara channelPara{
        commAddr, CCU_DEFAULT_REQUEST_CHANNEL_NUM, CCU_DEFAULT_REQUEST_JETTY_NUM, actualSqSize};
    std::vector<CcuChannelInfo> channelInfos;
    auto ret = CcuAllocChannels(devLogicId_, channelPara, channelInfos);
    CHK_PRT_RET(
        ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING(
            "[CcuChannelCtxPool][%s] failed to alloc ccu channels, ccu resources "
            "are unavailable, locAddr[%s] devLogicId[%d].",
            __func__, batchKey.Describe().c_str(), devLogicId_),
        ret);
    CHK_RET(ret);
    // 如果新增资源保存失败，手动释放避免泄露
    ret = CreateAndSaveNewBatch(batchKey, channelInfos, batchPtr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuChannelCtxPool][%s] failed, try to release temp ccu resources, locAddr[%s], "
            "devLogicId[%d], .",
            __func__, batchKey.Describe().c_str(), devLogicId_);
        for (const auto& channelInfo : channelInfos) {
            const auto dieId = channelInfo.dieId;
            const auto channelId = channelInfo.channelId;
            CHK_RET(CcuReleaseChannel(devLogicId_, dieId, channelId));
        }
        return ret;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::CreateAndSaveNewBatch(
    const BatchKey& batchKey, const std::vector<CcuChannelInfo> channelInfos, ResourceBatch*& batchPtr)
{
    // todo: 需要检查资源管理是否存在泄露可能
    auto& batches = batchMap_[batchKey];
    std::unique_ptr<ResourceBatch> newBatch{nullptr};
    newBatch.reset(new (std::nothrow) ResourceBatch(batchKey));
    CHK_PTR_NULL(newBatch);
    CHK_RET(newBatch->Init(channelInfos));
    for (const auto& channelInfo : channelInfos) {
        const auto dieId = channelInfo.dieId;
        const auto channelIdKey = std::make_pair(dieId, channelInfo.channelId);

        std::vector<CcuJetty*> jettys;
        for (const auto& jettyInfo : channelInfo.jettyInfos) {
            const auto jettyIdKey = std::make_pair(dieId, jettyInfo.taJettyId);
            jettys.emplace_back(newBatch->jettys[jettyIdKey].get());
        }

        channelJettyInfoMap_.emplace(channelIdKey, std::make_pair(channelInfo, jettys));
        usedChannelCntMap_[dieId] += 1;
    }

    batches.push_back(std::move(newBatch));
    ResourceBatch* rawBatch = batches.back().get();

    unconfirmedRecord_.newBatchSet.insert(rawBatch);
    batchPtr = rawBatch;
    return HcclResult::HCCL_SUCCESS;
}

bool CcuChannelCtxPool::FindAvailableBatch(const BatchKey& batchKey, ResourceBatch*& batchPtr) const
{
    auto it = batchMap_.find(batchKey);
    if (it == batchMap_.end()) {
        return false;
    }

    auto& batches = it->second;
    if (batches.empty()) {
        return false;
    }
    // 当前分配逻辑只有最后一个batch可能还有空闲资源
    auto& lastBatch = batches.back();
    if (lastBatch->availableChannelIdKeys.empty()) {
        return false;
    }

    batchPtr = lastBatch.get();
    return true;
}

HcclResult
CcuChannelCtxPool::GetChannelCtx(const Hccl::LinkData& link, CcuChannelCtxPool::CcuChannelCtx& channelCtx) const
{
    const auto& it = allocatedChannelIdMap_.find(link);
    CHK_PRT_RET(
        it == allocatedChannelIdMap_.end(),
        HCCL_ERROR(
            "[CcuChannelCtxPool][%s] failed to find allocated channelId of link[%s], ", "devLogicId[%d].", __func__,
            link.Describe().c_str(), devLogicId_),
        HcclResult::HCCL_E_NOT_FOUND);
    // 内部维护数据保证channelJettyInfoMap_记录的资源存在
    channelCtx = channelJettyInfoMap_.at(it->second);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::ReleaseConfirmedChannelRes()
{
    for (const auto& infoEntry : channelJettyInfoMap_) {
        const auto& channelIdKey = infoEntry.first;
        const auto dieId = channelIdKey.first;
        const auto channelId = channelIdKey.second;
        CHK_RET(CcuReleaseChannel(devLogicId_, dieId, channelId));
    }
    isReleased_ = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::GetCcuChannelCtxById(const std::pair<uint8_t, uint32_t>& key, CcuChannelCtx& ctx)
{
    auto it = channelJettyInfoMap_.find(key);
    if (it == channelJettyInfoMap_.end()) {
        HCCL_ERROR("[%s]fail, key[%u, %u] not found", __func__, key.first, key.second);
        return HCCL_E_NOT_FOUND;
    }
    ctx = it->second;
    return HCCL_SUCCESS;
}
} // namespace hcomm
