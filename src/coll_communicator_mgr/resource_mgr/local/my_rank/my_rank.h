/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MY_RANK_H
#define MY_RANK_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "hccl/hccl_res.h"
#include "hccl_mem_defs.h"
#include "acl/acl_rt.h"
#include "socket_manager.h"
#include "hcomm_res_defs.h"
#include "mem_host_pub.h"
#include "rank_pair_mgr.h"
#include "endpoint_mgr.h"
#include "comm_config_pub.h"
#include "manager_common.h"
#include "common.h"
#include "comm_mems/comm_mems.h"
#include "engine_ctxs.h"
#include "../../../dfx/ns_recovery/ns_recovery.h"
#include "hdc_pub.h"
#include "rank_graph.h"
#include "orion_adapter_hccp.h"
#include "coll_comm_config_consistency.h"
#include "exchange_info_mgr.h"

#include "ccu_types.h"
#include "ccu_drv_handle.h"
#include "dev_type.h"

namespace hccl {

/**
 * @note 职责：管理当前通信域下本Rank的信息和通信资源
 */
class MyRank {
public:
    MyRank(
        aclrtBinHandle binHandle, uint32_t rankId, const CommConfig& config, const ManagerCallbacks& callbacks,
        RankGraph* rankGraph, const Hccl::RankIpPortMapPtr& rankIpPortMap);
    ~MyRank();

    HcclResult Init(HcclMem cclBuffer, const uint32_t opExpansionMode, uint32_t rankNum);

    CommMems* GetCommMems() const { return commMems_.get(); }

    EngineCtxs* GetEngineCtxs() const { return engineCtxs_.get(); }

    uint32_t GetOpExpansionMode() { return opExpansionMode_; }
    CcuInsHandle GetCcuInstance() const { return ccuInsHandle_; }
    void SetCcuInstance(CcuInsHandle ccuInsHandle) { ccuInsHandle_ = ccuInsHandle; }

    CollCommConfigConsistency& GetCollCommConfigConsistency();

    hcomm::EndpointMgr* GetEndpointMgr() const { return endpointMgr_.get(); }

    HcclResult CreateChannels(
        CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs, uint32_t channelNum,
        ChannelHandle* channels);

    HcclResult ChannelGetHcclBuffer(ChannelHandle channel, void** buffer, uint64_t* size);
    HcclResult
    ChannelGetRemoteMems(ChannelHandle channel, uint32_t* memNum, CommMem** remoteMem, char*** memTags) const;
    HcclResult ChannelGetRemoteMems(
        ChannelHandle channel, uint32_t* memNum, CommMem** remoteMem, std::vector<std::string>& memTags) const;

    // Ns recovery
    void SetKfcControlTransfer(
        std::shared_ptr<HDCommunicate> kfcControlTransferH2D, std::shared_ptr<HDCommunicate> kfcStatusTransferD2H);
    std::vector<ChannelHandle> GetAllChannelList();
    HcclResult StopLaunch();
    HcclResult Clean();
    HcclResult Resume();

    /**
     * @brief 批量预建 socket（server 监听 + client 连接），与非共享路径 MyRank::CreateChannels 一致。
     *        共享 jetty 路径在 createFunc 中调用，确保两端 socket 体系一致、可对接。
     * @param[in] channelDescs HCCL 层 channel desc 数组
     * @param[in] channelNum 数量
     * @param[in] socketTag socket 标签（commTag + "_engine_" + engine）
     * @param[out] hcommDescs 输出的 HcommChannelDesc 数组（调用前需用 ChannelDescHccl2Hcomm 填充基础字段，
     *             本方法补上 socket/role/port 字段）
     */
    HcclResult BatchCreateSockets(
        const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
        std::vector<HcommChannelDesc>& hcommDescs);

    /**
     * @brief 在已建好的 socket 上执行通信域一致性校验交换（CheckFrameV2 + 用户信息）。
     *        与非共享路径 MyRank::CreateChannels 内部调用 exchangeInfoMgr_ 的逻辑一致，
     *        供共享 jetty 路径在 createFunc 中调用，确保两端无论 shared/非shared 配置是否对称，
     *        都会在 socket 上完成定长 CheckFrameV2（120B）的对称收发，避免一端死等。
     * @param[in] channelDescs HCCL 层 channel desc 数组
     * @param[in] hcommDescs 已通过 BatchCreateSockets 填充 socket 字段的 HcommChannelDesc 数组
     * @param[in] channelNum 数量
     * @param[in] newChannels 新建 channel 的 (idx, reuseIdx) 列表；空表示全部按新建处理
     * @param[in] engine 通信引擎
     * @note 仅 DEV_TYPE_950 实际执行交换，其它设备类型直接返回 SUCCESS，与非共享路径保持一致。
     */
    HcclResult BatchExchangeAndCheckConsistency(
        const HcclChannelDesc* channelDescs, const std::vector<HcommChannelDesc>& hcommDescs, uint32_t channelNum,
        const std::vector<std::pair<u32, u32>>& newChannels, CommEngine engine);

private:
    using ReuseSocketIdxMap = std::unordered_map<RankPair*, std::unordered_map<hcomm::EndpointPair*, u32>>;
    HcclResult GetEndpointPairFromChannel(
        const HcclChannelDesc& channelDesc, uint32_t channelIndex, uint32_t channelNum, uint32_t& remoteRank,
        hcomm::EndpointPair*& endpointPair, RankPair*& rankPair);
    HcclResult BatchServerInitForChannels(
        const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
        ReuseSocketIdxMap& reuseSocketIdxMap);
    HcclResult BatchGetSocketsForChannels(
        const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
        std::vector<HcommChannelDesc>& hcommDescs, ReuseSocketIdxMap& reuseSocketIdxMap);
    HcclResult BatchCreateChannels(
        CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
        std::vector<HcommChannelDesc>& hcommDescs, ChannelHandle* channelHandles);
    HcclResult
    BatchConnectChannels(const HcclChannelDesc* channelDescs, ChannelHandle* channelHandles, uint32_t channelNum);
    HcclResult CheckChannelParam(CommEngine engine, const HcclChannelDesc* channelDesc, uint32_t channelNum) const;
    HcclResult QueryListenPort(
        uint32_t localRank, uint32_t remoteRank, const EndpointDesc& localEndpointDesc,
        const EndpointDesc& remoteEndpointDesc, uint32_t& listenPort, HcommChannelDesc& hcommDesc);
    HcclResult GetLocalTlsStatus(Hccl::TlsStatus& tlsStatus) const;

    HcclResult TryInitCcuInstanceLegacy();
    HcclResult TryInitCcuInstance();
    HcclResult ReserveCcuMsCommOrFallback();
    HcclResult TryInitCcuInstanceOnDemand();
    void ReconcileCcuMsCommReservation(HcclResult initRet);
    void ReleaseCcuMsCommReservation();
    HcclResult ConfigSqDepthByExpansionMode(CommEngine engine, HcommChannelDesc& hcommDesc) const;
    HcclResult DestroyNewChannels(CommEngine engine, const HcclChannelDesc* channelDescs);
    // 获取port
    HcclResult GetListenPortInternal(uint32_t rank, uint32_t* devPort, EndpointLocType locType);

    aclrtBinHandle binHandle_{nullptr};
    uint32_t rankId_{};
    int32_t devLogicId_{};
    CommConfig config_{};

    // 当前通信域初始化没有处理CommConfig，暂时只使用展开模式
    uint32_t opExpansionMode_{0};

    std::unique_ptr<RankPairMgr> rankPairMgr_{nullptr};
    std::unique_ptr<hcomm::EndpointMgr> endpointMgr_{nullptr};
    std::unique_ptr<CommMems> commMems_{nullptr};
    std::unique_ptr<EngineCtxs> engineCtxs_{nullptr};

    CcuInsHandle ccuInsHandle_{0};

    ManagerCallbacks callbacks_;

    // RankGraph (临时放在myRank里面，后面会随着createchannel整体迁移到RankPairMgr上)
    RankGraph* rankGraph_{nullptr};

    // 记录每次调用BatchCreateChannels时新增的channelIndex, reuseIdx
    std::vector<std::pair<u32, u32>> newChannels_{};

    // Ns recovery
    std::unique_ptr<NsRecoveryProcessor> nsRecoveryProcessor_{nullptr};
    // 内部获取 port 的方法，根据 mode_ 区分 v1/v2
    HcclResult GetDevicePortInternal(uint32_t rank, uint32_t* devPort, EndpointLocType locType);

    Hccl::RankIpPortMapPtr rankIpPortMap_;

    CollCommConfigConsistency collCommConfigConsistency_;
    ExchangeInfoMgr exchangeInfoMgr_;
    std::shared_ptr<hcomm::CcuDrvHandle> ccuDrvHandle_{};
    bool useCcuResStaticAlloc_{false}; // HCCL版本不支持CCU资源按需申请
    bool ccuMsCommReserved_{false};
    Hccl::HcclMainboardId mainBoardType_{Hccl::HcclMainboardId::MAINBOARD_OTHERS};
};

} // namespace hccl

namespace MyRankUtils {

HcommChannelDesc ChannelDescHccl2Hcomm(const HcclChannelDesc& hcclDesc, const hccl::CommConfig& commConfig);

} // namespace MyRankUtils

#endif // MY_RANK_H
