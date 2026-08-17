/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_COMP_H
#define CCU_COMP_H

#include <array>
#include <stack>
#include <mutex>
#include <memory>
#include <vector>
#include <cstdint>
#include <utility>
#include <unordered_map>

#include "hccl_types.h"

#include "tp_mgr.h"
#include "ccu_dev_mgr_imp.h"
#include "ccu_res_allocator.h"
#include "ccu_res_specs.h"
#include "ccu_channel_ctx_mgr.h"

// 暂时引入orion仓
#include "local_ub_rma_buffer.h"

namespace hcomm {

class CcuComponent {
public:
    static CcuComponent& GetInstance(const int32_t deviceLogicId);
    HcclResult Init();
    HcclResult Deinit();

    HcclResult GetCcuResourceSpaceBufInfo(const uint8_t dieId, uint64_t& addr, uint64_t& size) const;
    HcclResult GetCcuResourceSpaceTokenInfo(const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue) const;

    HcclResult
    AllocChannels(const uint8_t dieId, const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos);
    HcclResult ConfigChannel(const uint8_t dieId, const ChannelCfg& cfg);
    HcclResult ReleaseChannel(const uint8_t dieId, const uint32_t channelId);

    HcclResult GetLoopChannelId(const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channelId) const;

    HcclResult AllocRes(
        const uint8_t dieId, const ResType resType, const uint32_t num, const bool consecutive,
        std::vector<ResInfo>& resInfos);
    HcclResult ReleaseRes(const uint8_t dieId, const ResType resType, const uint32_t startId, const uint32_t num);

    HcclResult AllocIns(const uint8_t dieId, const uint32_t num, ResInfo& insInfo);
    HcclResult ReleaseIns(const uint8_t dieId, const ResInfo& insInfo);
    uint32_t GetInsConsecutiveRemainSize(const uint8_t dieId) const;
    HcclResult AllocCke(const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos);
    HcclResult ReleaseCke(const uint8_t dieId, const std::vector<ResInfo>& ckeInfos);
    HcclResult AllocXn(const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos);
    HcclResult ReleaseXn(const uint8_t dieId, const std::vector<ResInfo>& xnInfos);

    // 0.5rtt专用接口
    HcclResult AllocWishCntXn(const uint8_t dieId, const std::string& resGroupTag, uint32_t& wishCntXn);
    HcclResult ReleaseWishCntXn(const uint8_t dieId, const std::string& resGroupTag, uint32_t wishCntXn);
    HcclResult
    GetCntXnBlock(const uint8_t dieId, const std::string& resGroupTag, std::pair<uint32_t, uint32_t>& cntXnPair);
    HcclResult GetTotalCntXn(const uint8_t dieId, const std::string& resGroupTag, uint32_t& totalCntXn);

    const std::array<bool, CCU_MAX_IODIE_NUM>& GetDieEnableFlags() const;

    HcclResult CleanTaskKillState() const;
    HcclResult CleanDieCkes(const uint8_t dieId) const;
    HcclResult CcuCleanTaskKillState(const int32_t deviceLogicId);
    HcclResult SetTaskKillDone();
    HcclResult SetTaskKill();

private:
    explicit CcuComponent() = default;
    ~CcuComponent();
    CcuComponent(const CcuComponent& that) = delete;
    CcuComponent& operator=(const CcuComponent& that) = delete;

    HcclResult CheckDiesEnable();
    HcclResult ChooseLoopEids(const std::array<bool, CCU_MAX_IODIE_NUM>& dieDrvEnableFlags);
    HcclResult GetLoopFeIpByDieId(const uint8_t dieId, uint32_t& feId, CommAddr& commAddr);
    HcclResult CreateCcuRmaBuffer();
    HcclResult CreateResourceManagers();
    HcclResult CreateLoopChannels();
    HcclResult CreateLoopChannel(const uint8_t dieId, uint32_t& channelId);
    HcclResult
    CreateAndImportLoopJettys(const uint8_t dieId, const CommAddr& commAddr, const std::vector<JettyInfo>& jettyInfos);
    HcclResult GetLoopTpInfo(const uint8_t dieId, const CommAddr& commAddr, TpInfo& tpInfo);
    HcclResult GetLoopTpAttr(const uint8_t dieId, const CommAddr& commAddr, TpAttrInfo& tpAttrInfo);
    uint32_t GetNewPsn();
    HcclResult ConfigLoopChannel(const uint8_t dieId, const CommAddr& commAddr, const ChannelInfo& channelInfo);
    HcclResult ConfigMsIdToken();

    HcclResult ReleaseJettyRes();
    HcclResult UnimportAllJettys();
    HcclResult ReleaseAllTpInfos();
    HcclResult DestroyAllJettys();

    HcclResult SetProcess(CcuOpcodeType opCode) const;
    HcclResult CcuSetTaskKillDone(const int32_t deviceLogicId);

    // 0.5rtt专用接口
    HcclResult ConfirmCntXns(const uint8_t dieId, const std::string& resGroupTag, const ResInfo& cntXnInfos);
    HcclResult GetAvailableTotalCntXnIndex(uint32_t& index) const;
    HcclResult
    SetTotalCntXnProcess(uint8_t dieId, uint32_t index, uint32_t fromId, uint32_t toId, uint32_t totalId) const;

    HcclResult SetSplitUnit(uint8_t dieId, uint32_t splitPktUnit) const;
    HcclResult SetTotalCntXn(uint8_t dieId, uint32_t fromId, uint32_t toId, uint32_t totalId, uint32_t index);
    HcclResult ResetTotalCntXn(uint8_t dieId, uint32_t index);

private:
    std::mutex innerMutex_;
    std::mutex taskKillMutex_;
    static constexpr uint32_t INVALID_DEV_ID = 0xFFFFFFFF;
    bool initFlag_{false};
    int32_t devLogicId_{static_cast<int32_t>(INVALID_DEV_ID)};
    uint32_t devPhyId_{INVALID_DEV_ID};
    CcuVersion ccuVersion_{CcuVersion::CCU_INVALID};

    // 根据资源规格的记录可用的die，要求drv可用，且环回eid存在
    std::array<bool, CCU_MAX_IODIE_NUM> dieEnableFlags_{};

    // 记录环回设备信息，dieId, (feId, commAddr)
    std::unordered_map<uint8_t, std::pair<uint32_t, CommAddr>> loopFeCommAddrMap_{};
    // 记录CCU资源空间Buffer，避免重复内存注册
    std::unordered_map<uint8_t, std::unique_ptr<Hccl::LocalUbRmaBuffer>> ccuRmaBufferMap_{};
    // 资源管理器
    std::array<std::unique_ptr<CcuChannelCtxMgr>, CCU_MAX_IODIE_NUM> channelCtxMgrs_{};
    std::array<std::unique_ptr<CcuResAllocator>, CCU_MAX_IODIE_NUM> resAllocators_{};
    // 环回channel编号
    static constexpr uint16_t INVAILD_LOOP_CHANNEL_ID = 0xFFFF;
    std::array<uint32_t, CCU_MAX_IODIE_NUM> loopChannelIds_{INVAILD_LOOP_CHANNEL_ID, INVAILD_LOOP_CHANNEL_ID};
    // 环回jetty资源信息
    // std::array<HcclNetDev, CCU_MAX_IODIE_NUM> netDevs_{}; 当前netdevs不支持jfc
    std::unordered_map<uint8_t, std::vector<HrtRaUbJettyCreatedOutParam>> createdOutParamMap_{};
    using ImportOutParamPair = std::pair<CtxHandle, HrtRaUbJettyImportedOutParam>;
    std::unordered_map<uint8_t, std::vector<ImportOutParamPair>> importedOutParamMap_{};
    std::unordered_map<uint8_t, TpInfo> tpInfoMap_{};
    std::unordered_map<uint8_t, TpAttrInfo> tpAttrInfoMap_{};
    enum class CcuTaskKillStatus : uint8_t { INIT = 0, TASK_KILL = 1, KILL_DONE = 2, CLEAN_TIF = 3, INVALID = 4 };
    CcuTaskKillStatus status{CcuTaskKillStatus::INVALID};

    struct CntXnBlock {
        ResInfo resInfo{};               // cntXn resInfo
        std::stack<uint32_t> wishCntXns; // wishCntXn Id
        uint32_t totalCntXn{0};          // totalCntXn Id
        uint32_t blockIdx{0};            // wishCntXn和totalCntXn绑定时的idx
    };
    std::mutex cntXnBlockMutex_;
    std::unordered_map<uint8_t, std::unordered_map<std::string, struct CntXnBlock>>
        cntXnBlocks_; // {dieId, {resGroupTag, CntXnBlock}}
    // 已使用的0.5RTT配置寄存器的index
    std::array<bool, CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM> usedTotalCntXnFlags_{};
};

} // namespace hcomm
#endif // CCU_COMP_H
