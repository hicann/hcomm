/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm.h"
#include "exception_handler.h"
#include "rank_graph_v2.h"
#include "kfc.h"
#include "dlhal_function.h"
#include "hcclCommTaskException.h"
#include "hccl_team_mgr.h"
#include "hccl_team_c_adpt.h"
#include "hcomm_team.h"
#include "hcomm_team_c_adpt.h"
#include "hccl/hccl_channel.h"
#include "hccl/hccl_rank_graph.h"

#include <cstdint>
#include <exception>
#include <unordered_set>
#include "launch_aicpu.h"
#include "launch_device.h"

namespace hccl {
void SymmetricMemoryDeleter::operator()(SymmetricMemory* ptr) const { delete ptr; }

CollComm::CollComm(
    void* comm, uint32_t rankId, const std::string& commName, const ManagerCallbacks& callbacks,
    CollCommInitMode initMode)
    : comm_(comm),
      rankId_(rankId),
      commId_(commName),
      config_(commName),
      callbacks_(callbacks),
      initMode_(initMode)
{
    groupScheduleMgr = std::make_shared<GroupScheduleMgr>();
}

CollComm::~CollComm()
{
    if (!IsFullMode()) { // SimpleMode是简化版collComm，只初始化了myrank、rankgraph未初始化以下资源，不需要析构处理
        return;
    }

    // 先注销TaskException，再销毁通信域资源，防止通信域资源销毁后rts回调TaskException
    hcomm::TaskExceptionHost* handler = hcomm::TaskExceptionHost::GetInstance(deviceLogicId_);
    if (handler != nullptr) {
        (void)handler->UnRegister(reinterpret_cast<u64>(this));
    }

    CHK_PRT(HcclBinaryUnLoad());

    // 兜底释放所有team的syncMem本地内存
    HcclTeamMgr::GetInstance().ClearByCollComm(this);
    // 兜底释放所有未注销的 HcommWindow device 副本（legacySymWin 部分由 symmetricMemory_ 析构清理）
    {
        std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
        for (auto& pair : hcommToSymMap_) {
            (void)HcommTeamWindowDeregister(pair.first); // 释放 L3 回填资源（remoteMems/sizes）
        }
        hcommToSymMap_.clear();
        symToHcommMap_.clear();
    }
    HCCL_INFO("[CollComm][~CollComm] collComm deinit");
    (void)DestroyAicpuComm();
    HCCL_RUN_INFO("[CollComm][~CollComm] cclBuffer free, commId[%s].", commId_.c_str());
}

HcclResult CollComm::Init(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    if (IsFullMode()) { // A5和下一代
        return InitFullMode(rankGraph, binHandle, cclBuffer, opExpansionMode);
    } else { // A2/A3使用简化版CollComm
        return InitSimpleMode(rankGraph, binHandle, cclBuffer, opExpansionMode);
    }
}

HcclResult
CollComm::InitSimpleMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    CHK_PTR_NULL(rankGraph);

    EXCEPTION_HANDLE_BEGIN

    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());

    // SimpleMode: A2/A3的RankGraph是保存在hccl::Communicator中的静态对象裸指针，CollComm不负责释放
    rankgraph_ = static_cast<RankGraph*>(rankGraph);

    uint32_t rankNum = 0;
    CHK_PTR_NULL(rankgraph_);
    CHK_RET(rankgraph_->GetRankSize(&rankNum));

    EXCEPTION_CATCH(
        myRank_ = std::make_shared<MyRank>(binHandle, rankId_, config_, callbacks_, rankgraph_, rankIpPortMap_),
        return HCCL_E_PTR);

    CHK_RET(myRank_->Init(cclBuffer, opExpansionMode, rankNum));

    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult
CollComm::InitFullMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    CHK_PTR_NULL(rankGraph);

    EXCEPTION_HANDLE_BEGIN

    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    rankGraphOwner_ = std::make_unique<RankGraphV2>(rankGraph);
    rankgraph_ = rankGraphOwner_.get();
    uint32_t rankNum = 0;
    CHK_PTR_NULL(rankgraph_);
    CHK_RET(rankgraph_->GetRankSize(&rankNum));
    CHK_RET(GetRankIpPortMap());

    u32 threadNum = 0xffffffff;
    u32 notifyNumPerThread = 0xffffffff;
    if (!commEngineResMgr_) {
        EXCEPTION_CATCH(commEngineResMgr_ = std::make_unique<CommEngineResMgr>(), return HCCL_E_PTR);
        CHK_PRT(commEngineResMgr_->Init(threadNum, notifyNumPerThread, commId_, binHandle, callbacks_));
    }

    if (!contextMgr_) {
        EXCEPTION_CATCH(contextMgr_ = std::make_unique<ContextManager>(), return HCCL_E_PTR);
    }

    EXCEPTION_CATCH(
        myRank_ = std::make_shared<MyRank>(binHandle, rankId_, config_, callbacks_, rankgraph_, rankIpPortMap_),
        return HCCL_E_PTR);
    CHK_RET(myRank_->Init(cclBuffer, opExpansionMode, rankNum));
    CHK_RET(hrtGetDevice(&deviceLogicId_));
    CHK_RET(InitSymmetricMemory());
    CHK_RET(InitWorldTeams());

    CHK_RET(InitHDCommunicate());

    if (!hcclCommDfx_) {
        EXCEPTION_CATCH(hcclCommDfx_ = std::make_unique<HcclCommDfx>(), return HCCL_E_PTR);
    }
    CHK_RET(hcclCommDfx_->Init(deviceLogicId_, commId_, rankId_));
    CHK_RET(InitTaskExceptionHandler());

    CHK_RET(InitKfcAndRegisterCollComm());

    Hccl::HcclCommunicator* comV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(comV2);
    CHK_RET(comV2->GetCclBufferSharedPtr(cclBuffer_));

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitSymmetricMemory()
{
    uint32_t rankSize = GetRankSize();
    HCCL_RUN_INFO(
        "[CollComm][InitSymmetricMemory] commId[%s], rank[%u], rankSize[%u].", commId_.c_str(), rankId_, rankSize);

    EXCEPTION_CATCH(
        symmetricMemory_.reset(new SymmetricMemory(rankId_, rankSize, 0, SymmetricMemoryMode::URMA)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(symmetricMemory_);
    return HCCL_SUCCESS;
}

HcclResult
CollComm::CreatePrebuiltWorldTeam(CommProtocol protocol, uint32_t netLayer, const std::vector<uint32_t>& reachableRanks)
{
    HcommTeamHandle worldTeam = HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(this, protocol, netLayer);
    if (worldTeam != nullptr) {
        HCCL_INFO(
            "[CollComm][%s] worldTeam for protocol[%d] netLayer[%u] already exists, handle[%p]", __func__, protocol,
            netLayer, worldTeam);
        return HCCL_SUCCESS;
    }
    // 构建ranks，包含本rank的id
    uint32_t selfMemberId = 0;
    std::vector<uint32_t> ranks = reachableRanks;
    ranks.emplace_back(rankId_);
    std::sort(ranks.begin(), ranks.end());
    uint32_t rankNum = ranks.size();
    // 查找selfMemberId
    for (uint32_t i = 0; i < rankNum; ++i) {
        if (ranks[i] == rankId_) {
            selfMemberId = i;
            break;
        }
    }

    HcommTeamCreateDesc hcommDesc{};
    (void)HcommTeamCreateDescInit(&hcommDesc);
    hcommDesc.memberNum = rankNum;
    hcommDesc.selfMemberId = selfMemberId;
    hcommDesc.netLayer = netLayer;
    hcommDesc.protocol = protocol;
    hcommDesc.requirement.barrierCount = 1;

    HcommTeamHandle newTeam = nullptr;
    uint64_t syncMemSize = 0;
    HcommResult createRet = HcommTeamCreate(nullptr, &hcommDesc, &newTeam, &syncMemSize);
    CHK_PRT_RET(
        createRet != 0,
        HCCL_ERROR(
            "[CollComm][%s] HcommTeamCreate failed, protocol[%d] netLayer[%u] ret[%d]", __func__, protocol, netLayer,
            createRet),
        HCCL_E_INTERNAL);

    HcclResult regRet = HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(
        newTeam, this, protocol, netLayer, ranks.data(), rankNum);
    if (regRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CollComm][%s] RegisterPrebuiltWorldTeam failed, protocol[%d] netLayer[%u] ret[%d]", __func__, protocol,
            netLayer, regRet);
        (void)HcommTeamDestroy(newTeam);
        return regRet;
    }
    HCCL_INFO("[CollComm][%s] prebuilt worldTeam for protocol[%d] netLayer[%u]", __func__, protocol, netLayer);
    return HCCL_SUCCESS;
}

/* 单层可达 rank 收集：遍历本 rank 到该层所有 rank 的全部 link，按协议分别记录可达 rank（同协议多条
 * link 只记一次该 peer）。URMA 各协议与 UB_MEM 分别记录。 */
void CollComm::CollectLayerReachableRanks(
    uint32_t netLayer, const uint32_t* ranks, uint32_t rankNum,
    std::unordered_map<CommProtocol, std::vector<uint32_t>>& protoReachableRanks)
{
    for (uint32_t ri = 0; ri < rankNum; ++ri) {
        if (ranks[ri] == rankId_) {
            continue;
        }
        CommLink* links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = rankgraph_->GetLinks(netLayer, rankId_, ranks[ri], &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
            HCCL_INFO(
                "[CollComm][%s] netLayer[%u] selfRank[%u] remoteRank[%u] has no valid links, skip", __func__, netLayer,
                rankId_, ranks[ri]);
            continue;
        }
        // 两rank之间可能存在多条不同协议的link，需遍历全部；同一协议只记一次该peer
        for (uint32_t lk = 0; lk < linkNum; ++lk) {
            CommProtocol proto = links[lk].linkAttr.linkProtocol;
            auto it = protoReachableRanks.find(proto);
            if (it == protoReachableRanks.end()) {
                continue;
            }
            if (std::find(it->second.begin(), it->second.end(), ranks[ri]) == it->second.end()) {
                it->second.emplace_back(ranks[ri]);
            }
        }
    }
}

HcclResult CollComm::InitWorldTeams()
{
    CHK_PTR_NULL(rankgraph_);

    // 1. 获取所有netLayer
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(rankgraph_->GetNetLayers(&netLayers, &netLayerNum));
    CHK_PRT_RET(
        netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[CollComm][%s] GetNetLayers returned empty, commId[%s]", __func__, commId_.c_str()),
        HCCL_E_INTERNAL);

    // UB_MEM最高层信息（正序遍历时后覆盖前，最终得到最高层）
    uint32_t ubMemNetLayer = 0;
    std::vector<uint32_t> ubMemReachableRanks;

    // 2. 逐netLayer遍历，记录可达rank
    for (uint32_t li = 0; li < netLayerNum; ++li) {
        uint32_t netLayer = netLayers[li];
        uint32_t* ranks = nullptr;
        uint32_t rankNum = 0;
        CHK_RET(rankgraph_->GetInstRanksByNetLayer(netLayer, &ranks, &rankNum));
        if (ranks == nullptr || rankNum <= 1) {
            HCCL_INFO("[CollComm][%s] netLayer[%u] has no valid ranks, rankNum[%u], skip", __func__, netLayer, rankNum);
            continue;
        }

        // 各协议可达rank列表
        std::unordered_map<CommProtocol, std::vector<uint32_t>> protoReachableRanks
            = {{COMM_PROTOCOL_UB_CTP, {}},
               {COMM_PROTOCOL_UBC_TP, {}},
               {COMM_PROTOCOL_UBOE, {}},
               {COMM_PROTOCOL_UB_RTP, {}},
               {COMM_PROTOCOL_UB_MEM, {}}};
        CollectLayerReachableRanks(netLayer, ranks, rankNum, protoReachableRanks);

        // 创建URMA worldTeam
        for (auto& [proto, reachableRanks] : protoReachableRanks) {
            if (proto == COMM_PROTOCOL_UB_MEM || reachableRanks.size() == 0) {
                continue;
            }
            CHK_RET(CreatePrebuiltWorldTeam(proto, netLayer, reachableRanks));
        }

        // 记录UB_MEM最高层信息（正序遍历，后覆盖前，最终得到最高层）
        if (protoReachableRanks[COMM_PROTOCOL_UB_MEM].size() > 0) {
            ubMemNetLayer = netLayer;
            ubMemReachableRanks = protoReachableRanks[COMM_PROTOCOL_UB_MEM];
        }
    }

    // 3. UB_MEM：在记录的最高层预制1个worldTeam
    if (ubMemReachableRanks.size() > 0) {
        CHK_RET(CreatePrebuiltWorldTeam(COMM_PROTOCOL_UB_MEM, ubMemNetLayer, ubMemReachableRanks));
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::RegisterSymmetricMemoryResource(void* ptr, size_t size, SymmetricMemoryResource& resource)
{
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(
        size == 0, HCCL_ERROR("[CollComm][RegisterSymmetricMemoryResource] invalid symmetric memory size 0."),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(myRank_);

    CommMems* commMems = myRank_->GetCommMems();
    CHK_PTR_NULL(commMems);

    CommMem commMem{};
    commMem.type = COMM_MEM_TYPE_DEVICE;
    commMem.addr = ptr;
    commMem.size = static_cast<uint64_t>(size);
    resource.memTag = std::string(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) + commId_ + "_addr_"
                      + std::to_string(reinterpret_cast<uintptr_t>(ptr)) + "_size_" + std::to_string(size);
    HcclResult ret = commMems->CommRegMem(resource.memTag, commMem, &resource.memHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollComm][RegisterSymmetricMemoryResource] CommRegMem failed, tag[%s], ptr[%p], "
            "size[%zu], ret[%d].",
            resource.memTag.c_str(), ptr, size, ret),
        ret);

    HCCL_RUN_INFO(
        "[CollComm][RegisterSymmetricMemoryResource] register symmetric memory success, group[%s], "
        "tag[%s], ptr[%p], size[%zu], memHandle[%p].",
        commId_.c_str(), resource.memTag.c_str(), ptr, size, resource.memHandle);
    return HCCL_SUCCESS;
}

void CollComm::UnregisterSymmetricMemoryResource(const SymmetricMemoryResource& resource)
{
    if (resource.memHandle == nullptr || resource.memTag.empty()) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] invalid resource, tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    if (myRank_ == nullptr) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] myRank is null, skip CommUnregMem, "
            "tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    CommMems* commMems = myRank_->GetCommMems();
    if (commMems == nullptr) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] commMems is null, skip CommUnregMem, "
            "tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    HcclResult ret = commMems->CommUnregMem(resource.memTag, resource.memHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] CommUnregMem failed, tag[%s], "
            "memHandle[%p], ret[%d].",
            resource.memTag.c_str(), resource.memHandle, ret);
    }
    ret = myRank_->UnregMemByTag(resource.memTag);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CollComm][UnregisterSymmetricMemoryResource] UnregMemByTag failed, tag[%s], ret[%d].",
            resource.memTag.c_str(), ret);
        return;
    }
    HCCL_INFO(
        "[CollComm][UnregisterSymmetricMemoryResource] unregister symmetric memory success, "
        "tag[%s], memHandle[%p].",
        resource.memTag.c_str(), resource.memHandle);
}

/* 单 team 补交换：对每个 peer 建 1 条 channel（CreateChannels 内部有复用逻辑），symm memHandle 挂到
 * channelDesc 参与交换，建链后 UpdateSymmetricRemoteMem 双回填 SymmetricWindow/HcommWindow。 */
HcclResult CollComm::ReExchangeChannelsForTeam(
    HcommTeamHandle team, CommEngine engine, uint32_t netLayer, std::vector<HcclMemHandle>& symMemHandles)
{
    ChannelsCreateCtx ctx{};
    CHK_RET(GetTeamMemberInfo(team, rankId_, ctx));
    ctx.channelsByMember.assign(ctx.memberNum, {});
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        if (m == ctx.selfMemberId) {
            continue; // 不建到自己的 channel
        }
        uint32_t peerRank = ctx.rankIds[m];
        HcclChannelDesc channelDesc = {};
        HcclResult ret = HcclChannelDescInit(&channelDesc, 1);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[CollComm][%s] HcclChannelDescInit failed, ret[%d]", __func__, ret), ret);
        CommLink* links = nullptr;
        uint32_t linkNum = 0;
        ret = rankgraph_->GetLinks(netLayer, rankId_, peerRank, &links, &linkNum);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || links == nullptr || linkNum == 0,
            HCCL_ERROR("[CollComm][%s] no link from rank[%u] to rank[%u]", __func__, rankId_, peerRank),
            HCCL_E_NOT_FOUND);
        channelDesc.remoteRank = peerRank;
        channelDesc.channelProtocol = links[0].linkAttr.linkProtocol;
        channelDesc.localEndpoint = links[0].srcEndpointDesc;
        channelDesc.remoteEndpoint = links[0].dstEndpointDesc;
        channelDesc.memHandles = symMemHandles.data(); // 外部已判空，此处确保有内容
        channelDesc.memHandleNum = static_cast<uint32_t>(symMemHandles.size());
        ctx.channelsByMember[m].assign(1, 0);
        ret = myRank_->CreateChannels(engine, commId_, &channelDesc, 1, ctx.channelsByMember[m].data());
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollComm][%s] CreateChannels failed, peerRank[%u] ret[%d]", __func__, peerRank, ret), ret);

        // 从新 channel 取回交换到的 remoteMem 并回填（SymmetricWindow + HcommWindow）
        CommMem* remoteMems = nullptr;
        uint32_t memNum = 0;
        std::vector<std::string> memTags;
        CHK_RET(myRank_->ChannelGetRemoteMems(ctx.channelsByMember[m][0], &memNum, &remoteMems, memTags));
        if (memNum > 0) {
            CHK_RET(UpdateSymmetricRemoteMem(peerRank, remoteMems, memTags));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::ReExchangeWindowsForBoundTeams()
{
    // 已建链 subTeam（无则常规时序 window 先于 Team，无需补交换）
    std::vector<HcommTeamHandle> subTeams = HcclTeamMgr::GetInstance().GetLinkedSubTeams(this);
    if (subTeams.empty()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(rankgraph_);

    for (HcommTeamHandle team : subTeams) {
        // 运行时查 team 的 netLayer/engine（建链成功时由 HcclTeamCreate 写入 L3）
        uint32_t netLayer = 0;
        HcommResult nlRet = HcommTeamGetNetLayer(team, &netLayer);
        CHK_PRT_RET(
            nlRet != 0,
            HCCL_ERROR("[CollComm][%s] HcommTeamGetNetLayer failed, team[%p] ret[%d]", __func__, team, nlRet),
            static_cast<HcclResult>(nlRet));
        CommEngine engine = COMM_ENGINE_RESERVED;
        HcommResult engineRet = HcommTeamGetEngine(team, &engine);
        CHK_PRT_RET(
            engineRet != 0,
            HCCL_ERROR("[CollComm][%s] HcommTeamGetEngine failed, team[%p] ret[%d]", __func__, team, engineRet),
            HCCL_E_INTERNAL);

        // 注册 pending window 的 memHandle（新注册 window 的 CommRegMem 延迟到此刻执行）
        CHK_RET(RegisterPendingSymmetricMemHandles());
        std::vector<HcclMemHandle> symMemHandles;
        CHK_RET(GetAllRegisteredSymMemHandles(symMemHandles));
        if (symMemHandles.empty()) {
            continue;
        }
        CHK_RET(ReExchangeChannelsForTeam(team, engine, netLayer, symMemHandles));
        HCCL_INFO("[CollComm][%s] reexchange done, team[%p]", __func__, team);
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::RegisterWindow(void* ptr, size_t size, HcclCommSymWindow* winHandle)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    CHK_PTR_NULL(winHandle);

    // 1. 创建 SymmetricWindow（AICPU URMA 路径），得到 devLegacySymWin（device 副本）
    //    后续失败路径 legacySymWin 部分由 symmetricMemory_ 析构清理
    void* devLegacySymWin = nullptr;
    CHK_RET(symmetricMemory_->RegisterUrmaSymmetricMem(ptr, size, &devLegacySymWin));

    // 2. 创建 HcommWindow device 副本
    void* devWin = nullptr;
    HcommResult regRet = HcommTeamWindowRegister(devLegacySymWin, &devWin);
    if (regRet != 0) {
        HCCL_ERROR("[CollComm][RegisterWindow] HcommTeamWindowRegister failed, ret[%d]", regRet);
        return HCCL_E_INTERNAL;
    }
    // 3. 登记 devHcommWindow↔devLegacySymWin 双向映射（供 DeregisterWindow 注销 SymmetricWindow、
    //    GetCommSymWin O(1) 反查），并登记到 L3 接管回填资源
    {
        std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
        hcommToSymMap_[devWin] = devLegacySymWin;
        symToHcommMap_[devLegacySymWin] = devWin;
    }

    // 3.5 登记本端窗口注册信息（用户 VA/size），供 HcommWindow.netWin.baseRemoteMemAddr 表回填本端槽位；
    //     本端层槽位此时未知，先置空，由 UpdateHcommWindowRemoteMem 首次调用时补齐
    HcommResult selfRet = HcommTeamWindowSetSelfInfo(devWin, ptr, size, nullptr, 0);
    if (selfRet != 0) {
        HCCL_ERROR("[CollComm][RegisterWindow] HcommTeamWindowSetSelfInfo failed, ret[%d]", selfRet);
        (void)HcommTeamWindowDeregister(devWin);
        {
            std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
            hcommToSymMap_.erase(devWin);
            symToHcommMap_.erase(devLegacySymWin);
        }
        return HCCL_E_INTERNAL;
    }

    *winHandle = devWin;

    // 4. window 后注册补交换：已存在建链 Team 时，带新 window 的 memHandle 重新建链交换并回填
    //    （无已建链 Team 的常规时序 window 先于 Team，内部为空操作）
    CHK_RET(ReExchangeWindowsForBoundTeams());

    HCCL_INFO(
        "[CollComm][RegisterWindow] success, HcommWindow[%p], devLegacySymWin[%p], ptr[%p], size[%zu]", devWin,
        devLegacySymWin, ptr, size);
    return HCCL_SUCCESS;
}

HcclResult CollComm::DeregisterWindow(HcclCommSymWindow winHandle)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);

    // 1. 从映射取出 devLegacySymWin（双向映射同步清理），注销 L3 回填资源并释放 HcommWindow device 副本
    void* devLegacySymWin = nullptr;
    {
        std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
        auto it = hcommToSymMap_.find(winHandle);
        CHK_PRT_RET(
            it == hcommToSymMap_.end(),
            HCCL_ERROR("[CollComm][DeregisterWindow] winHandle[%p] not found in hcommToSymMap_", winHandle),
            HCCL_E_NOT_FOUND);
        devLegacySymWin = it->second;
        hcommToSymMap_.erase(it);
        symToHcommMap_.erase(devLegacySymWin);
    }
    HcommResult unregRet = HcommTeamWindowDeregister(winHandle);
    if (unregRet != 0) {
        HCCL_WARNING("[CollComm][DeregisterWindow] HcommTeamWindowDeregister failed, ret[%d]", unregRet);
    }

    // 2. 注销 SymmetricWindow
    SymmetricMemoryResource resource;
    HcclResult getResourceRet = symmetricMemory_->GetRegisteredMemoryResource(devLegacySymWin, resource);
    // 清理 tagToHcommMap_ 中对应的 tag 条目
    if (getResourceRet == HCCL_SUCCESS && !resource.memTag.empty()) {
        std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
        tagToHcommMap_.erase(resource.memTag);
    } else if (getResourceRet != HCCL_SUCCESS && getResourceRet != HCCL_E_NOT_FOUND) {
        HCCL_WARNING(
            "[CollComm][DeregisterWindow] get registered symmetric memory resource failed, "
            "devLegacySymWin[%p], ret[%d].",
            devLegacySymWin, getResourceRet);
    }

    HcclResult ret = symmetricMemory_->DeregisterUrmaSymmetricMem(devLegacySymWin);
    if (ret == HCCL_SUCCESS && getResourceRet == HCCL_SUCCESS) {
        // 对称内存窗口注销后，同步删除本地memTag到memHandle索引。
        {
            std::unique_lock<std::shared_mutex> lock(registeredSymMemHandleMapMtx_);
            registeredSymMemHandleMap_.erase(resource.memTag);
        }
        UnregisterSymmetricMemoryResource(resource);
    }
    return ret;
}

HcclResult CollComm::GetCommSymWin(void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    // FindUrmaSymmetricWindow 返回 devLegacySymWin，经 symToHcommMap_ O(1) 反查 HcommWindow handle
    void* devLegacySymWin = nullptr;
    HcclResult ret = symmetricMemory_->FindUrmaSymmetricWindow(ptr, size, &devLegacySymWin, offset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[CollComm][GetCommSymWin] FindUrmaSymmetricWindow failed, ret[%d]", ret), ret);
    if (devLegacySymWin == nullptr) {
        // A5查询未命中不是错误，返回nullptr让算子侧按普通内存路径处理。
        *winHandle = nullptr;
        *offset = 0;
        return HCCL_SUCCESS;
    }
    {
        std::shared_lock<std::shared_mutex> lock(hcommWindowMutex_);
        auto symIt = symToHcommMap_.find(devLegacySymWin);
        CHK_PRT_RET(
            symIt == symToHcommMap_.end(),
            HCCL_ERROR("[CollComm][GetCommSymWin] no HcommWindow for devLegacySymWin[%p]", devLegacySymWin),
            HCCL_E_NOT_FOUND);
        *winHandle = symIt->second;
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::RegisterPendingSymmetricMemHandles()
{
    if (symmetricMemory_ == nullptr) {
        return HCCL_SUCCESS;
    }

    std::vector<SymmetricMemoryRegisterInfo> registerInfos;
    // HcclCommSymWinRegister只记录窗口，真正CommRegMem延迟到ChannelAcquire阶段执行。
    CHK_RET(symmetricMemory_->GetPendingRegisterInfos(registerInfos));
    if (registerInfos.empty()) {
        return HCCL_SUCCESS;
    }

    for (const SymmetricMemoryRegisterInfo& registerInfo : registerInfos) {
        SymmetricMemoryResource resource;
        HcclResult ret = RegisterSymmetricMemoryResource(registerInfo.userVa, registerInfo.userSize, resource);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][RegisterPendingSymmetricMemHandles] register symmetric memory failed, "
                "win[%p], userVa[%p], size[%zu], ret[%d].",
                registerInfo.devWin, registerInfo.userVa, registerInfo.userSize, ret);
            return ret;
        }

        ret = symmetricMemory_->SetRegisteredMemoryResource(registerInfo.devWin, resource);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][RegisterPendingSymmetricMemHandles] save symmetric memory resource failed, "
                "win[%p], userVa[%p], size[%zu], ret[%d].",
                registerInfo.devWin, registerInfo.userVa, registerInfo.userSize, ret);
            UnregisterSymmetricMemoryResource(resource);
            return ret;
        }
        // 注册成功后加入本地索引，供新建通道和后续复用通道使用。
        {
            std::unique_lock<std::shared_mutex> lock(registeredSymMemHandleMapMtx_);
            registeredSymMemHandleMap_[resource.memTag] = static_cast<HcclMemHandle>(resource.memHandle);
        }
        // 登记 memTag→HcommWindow 映射，供 UpdateHcommWindowRemoteMem 直接查
        {
            std::unique_lock<std::shared_mutex> lock(hcommWindowMutex_);
            // symToHcommMap_ 反查 devLegacySymWin 对应的 HcommWindow handle；查不到则不登记该 memTag
            auto symIt = symToHcommMap_.find(registerInfo.devWin);
            if (symIt != symToHcommMap_.end()) {
                tagToHcommMap_[resource.memTag] = symIt->second;
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CollComm::GetAllRegisteredSymMemHandles(std::vector<HcclMemHandle>& memHandles) const
{
    memHandles.clear();
    std::shared_lock<std::shared_mutex> lock(registeredSymMemHandleMapMtx_);
    for (const auto& registeredMem : registeredSymMemHandleMap_) {
        memHandles.emplace_back(registeredMem.second);
    }
    HCCL_INFO("[CollComm][GetAllRegisteredSymMemHandles] registeredMemHandleNum[%zu].", memHandles.size());
    return HCCL_SUCCESS;
}

HcclResult CollComm::GetRemoteMissingSymMemHandles(
    const std::vector<std::string>& remoteMemTags, std::vector<HcclMemHandle>& memHandles) const
{
    memHandles.clear();
    const std::unordered_set<std::string> remoteMemTagSet(remoteMemTags.begin(), remoteMemTags.end());
    std::shared_lock<std::shared_mutex> lock(registeredSymMemHandleMapMtx_);
    // 返回目标通道远端尚未拥有的本地句柄。
    for (const auto& registeredMem : registeredSymMemHandleMap_) {
        if (remoteMemTagSet.find(registeredMem.first) == remoteMemTagSet.end()) {
            memHandles.emplace_back(registeredMem.second);
        }
    }
    HCCL_INFO(
        "[CollComm][GetRemoteMissingSymMemHandles] remoteMemTagNum[%zu], missingMemHandleNum[%zu].",
        remoteMemTags.size(), memHandles.size());
    return HCCL_SUCCESS;
}

HcclResult CollComm::UpdateSymmetricRemoteMem(
    uint32_t remoteRank, const CommMem* remoteMems, const std::vector<std::string>& memTags)
{
    if (symmetricMemory_ == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_RET(symmetricMemory_->UpdateRemoteMem(remoteRank, remoteMems, memTags));
    // 同时回填 HcommWindow.netWin.baseRemoteMemAddr 偏移表（window 属于通信域，由 CollComm 统一回填）
    return UpdateHcommWindowRemoteMem(remoteRank, remoteMems, memTags);
}

HcclResult CollComm::UpdateHcommWindowRemoteMem(
    uint32_t remoteRank, const CommMem* remoteMems, const std::vector<std::string>& memTags)
{
    // L2 计算层分段信息：sizes 为各层 worldTeam 大小（变长，下标=netLayer），slots 为 remoteRank 在各层的最终槽位
    std::vector<uint32_t> sizes;
    HcclTeamMgr::GetInstance().GetWorldTeamSizesPerNetLayer(this, sizes);
    std::vector<std::pair<uint32_t, uint32_t>> layerSlots;
    HcclTeamMgr::GetInstance().GetRankLayerSlots(this, remoteRank, layerSlots);
    // 本端（self rank）层槽位：与 remoteRank 同一算法（sum(sizes[0..L-1]) + 该层 worldTeamId）。
    // 每个对端 rank 到达都会调一次本函数，这里登记是幂等的（重复写相同值无害）
    std::vector<std::pair<uint32_t, uint32_t>> selfLayerSlots;
    HcclTeamMgr::GetInstance().GetRankLayerSlots(this, rankId_, selfLayerSlots);
    std::vector<uint32_t> selfSlots;
    for (const auto& layerSlot : selfLayerSlots) {
        uint32_t offset = 0;
        for (uint32_t l = 0; l < layerSlot.first && l < sizes.size(); ++l) {
            offset += sizes[l];
        }
        selfSlots.emplace_back(offset + layerSlot.second);
    }
    std::vector<uint32_t> slots;
    for (const auto& layerSlot : layerSlots) {
        uint32_t offset = 0;
        for (uint32_t l = 0; l < layerSlot.first && l < sizes.size(); ++l) {
            offset += sizes[l];
        }
        slots.emplace_back(offset + layerSlot.second);
    }

    CHK_PTR_NULL(remoteMems);
    std::shared_lock<std::shared_mutex> lock(hcommWindowMutex_);
    for (size_t i = 0; i < memTags.size(); ++i) {
        if (memTags[i].empty()) {
            continue;
        }
        // 直接查 tagToHcommMap_ 得 HcommWindow handle（void*，与 HcclCommSymWindow 一致）
        auto it = tagToHcommMap_.find(memTags[i]);
        if (it == tagToHcommMap_.end()) {
            continue;
        }
        // 先补齐本端槽位（VA/size 已在 RegisterWindow 登记，这里只登记槽位；幂等）
        if (!selfSlots.empty()) {
            HcommResult selfRet = HcommTeamWindowSetSelfInfo(
                it->second, nullptr, 0, selfSlots.data(), static_cast<uint32_t>(selfSlots.size()));
            CHK_PRT_RET(
                selfRet != 0,
                HCCL_ERROR("[CollComm][UpdateHcommWindowRemoteMem] set self slots failed, ret[%d]", selfRet),
                static_cast<HcclResult>(selfRet));
        }
        HcommResult hRet = HcommTeamUpdateWindowRemoteMemByRank(
            it->second, sizes.data(), static_cast<uint32_t>(sizes.size()), slots.data(),
            static_cast<uint32_t>(slots.size()), &remoteMems[i]);
        CHK_PRT_RET(
            hRet != 0,
            HCCL_ERROR(
                "[CollComm][UpdateHcommWindowRemoteMem] failed, handle[%p] remoteRank[%u] ret[%d]", it->second,
                remoteRank, hRet),
            static_cast<HcclResult>(hRet));
    }
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitKfcAndRegisterCollComm()
{
    myRank_->SetKfcControlTransfer(kfcControlTransferH2D_, kfcStatusTransferD2H_);
    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    return HCCL_SUCCESS;
}

HcclResult CollComm::DestroyAicpuComm()
{
    CHK_PTR_NULL(callbacks_.getAicpuCommState);
    if (callbacks_.getAicpuCommState()) {
        CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
        CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);

        Hccl::KfcCommand opCmd = Hccl::KfcCommand::DESTROY_AICPU_COMM;
        CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
        HCCL_RUN_INFO(
            "[%s]group[%s] send Hccl::KfcCommand[%d] success", __func__, commId_.c_str(), static_cast<int>(opCmd));

        Hccl::KfcExecStatus opInfo;
        constexpr u32 WAIT_CMD_TIMEOUT = 10 * 1000; // 最大等待10秒
        auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(Hccl::KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
            if (opInfo.kfcStatus == Hccl::KfcStatus::DESTROY_AICPU_COMM_DONE) {
                HCCL_RUN_INFO("[%s]get Hccl::KfcStatus[%d] success", __func__, static_cast<int>(opInfo.kfcStatus));
                return HCCL_SUCCESS;
            } else if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                HCCL_ERROR(
                    "[%s]timeout, maxTime[%u ms] and get the opExecStatus is [%s].", __func__, WAIT_CMD_TIMEOUT,
                    opInfo.kfcStatus.Describe().c_str());
                return HCCL_E_TIMEOUT;
            }
            usleep(TEN_MILLISECOND_OF_USLEEP);
        }
    }
    return HCCL_SUCCESS;
}

uint32_t CollComm::GetMyRankId() const { return rankId_; }

HcclResult CollComm::GetParentRankId(u32& parentRankId) const
{
    Hccl::HcclCommunicator* comV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(comV2);
    parentRankId = comV2->GetRankInParentComm();
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitHDCommunicate()
{
    // 初始化aicpu进程 host-device 共享内存
    EXCEPTION_CATCH(
        (kfcControlTransferH2D_
         = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_H2D, sizeof(Hccl::KfcCommand))),
        return HCCL_E_PTR);
    CHK_RET(kfcControlTransferH2D_->InitHost());

    EXCEPTION_CATCH(
        (kfcStatusTransferD2H_
         = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_D2H, sizeof(Hccl::KfcExecStatus))),
        return HCCL_E_PTR);
    CHK_RET(kfcStatusTransferD2H_->InitHost());

    return HCCL_SUCCESS;
}

HcclResult CollComm::GetHDCommunicate(
    HDCommunicateParams& kfcControlTransferH2DParams, HDCommunicateParams& kfcStatusTransferD2HParams)
{
    CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
    CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
    kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();
    HCCL_INFO("%s success, group[%s]", __func__, commId_.c_str());
    return HCCL_SUCCESS;
}

HcclCommStatus CollComm::GetCommStatus() const
{
    std::lock_guard<std::mutex> lock(commMutex_);
    return commStatus_;
}

HcclResult CollComm::Suspend()
{
    HCCL_RUN_INFO("[CollComm][Suspend] commId[%s] start to suspend.", commId_.c_str());
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_WARNING("[CollComm][Suspend] The current communication has been suspended, no need to suspend again.");
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_SMART_PTR_NULL(myRank_);

        commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    }

    return myRank_->StopLaunch();
}

HcclResult CollComm::Clean()
{
    HCCL_RUN_INFO("[CollComm][Clean] commId[%s] start to clean.", commId_.c_str());
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ != HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_ERROR(
                "[CollComm][Clean] The current communication is not suspended, cannot clean, status is [%u]",
                static_cast<uint32_t>(commStatus_));
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        if (isCleaned_) {
            HCCL_WARNING("[CollComm][Clean] The current communication has been cleaned, no need to clean again.");
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_SMART_PTR_NULL(myRank_);

        isCleaned_ = true;
    }

    // 先清理Host
    return myRank_->Clean();
}

HcclResult CollComm::Resume()
{
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_INVALID) {
            HCCL_ERROR("[CollComm][Resume] Comm has been error, can not resume now!");
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (commStatus_ != HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_WARNING(
                "[CollComm][Resume] The current communication is normal, no need to resume, status is [%u]",
                static_cast<uint32_t>(commStatus_));
            return HcclResult::HCCL_SUCCESS;
        }

        HCCL_INFO("[CollComm][Resume] start to Resume.");
        CHK_SMART_PTR_NULL(myRank_);
        auto ret = myRank_->Resume();
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[CollComm][Resume] %s failed, ret = 0x%016llx", __func__, HCCL_ERROR_CODE(ret));
            return ret;
        }

        if (commEngineResMgr_ != nullptr) {
            auto notifyRet = commEngineResMgr_->ResetCommLocalNotifies();
            if (notifyRet != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[CollComm][Resume] ResetCommLocalNotifies failed, ret = 0x%016llx", HCCL_ERROR_CODE(notifyRet));
                return notifyRet;
            }
        }

        commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
        isCleaned_ = false;
    }
    HCCL_INFO("[CollComm][Resume] commId[%s] resume success.", commId_.c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollComm::InitTaskExceptionHandler()
{
    hcomm::TaskExceptionHost* handler = hcomm::TaskExceptionHost::GetInstance(deviceLogicId_);
    CHK_PTR_NULL(handler);
    CHK_RET(handler->Register(reinterpret_cast<u64>(this)));
    return HCCL_SUCCESS;
}

Hccl::ErrorMessageReport CollComm::GetAicpuTaskException()
{
    Hccl::ErrorMessageReport errorMessage;
    CHK_PRT_RET(kfcStatusTransferD2H_ == nullptr, HCCL_ERROR("[%s]fail, d2h is nullptr", __func__), errorMessage);

    HcclResult ret = kfcStatusTransferD2H_->Get(
        sizeof(Hccl::KfcStatus) + sizeof(Hccl::KfcErrType), sizeof(errorMessage),
        reinterpret_cast<uint8_t*>(&errorMessage));

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]fail, group [%s], ret[%d]", __func__, commId_.c_str(), static_cast<int>(ret)), errorMessage);
    HCCL_INFO("[%s]group[%s] success", __func__, commId_.c_str());
    return errorMessage;
}

uint32_t CollComm::UpdateIndex() { return index_ += 1; }

HcclResult CollComm::GetRankIpPortMap()
{
    Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(commV2);
    CHK_RET(commV2->GetRankIpPortMap(rankIpPortMap_));
    CHK_PTR_NULL(rankIpPortMap_);
    // rankIpPortMap_ 在单卡多进程场景下，用于保证端口不冲突
    // 该映射表记录了：Rank ID -> (IP地址 -> 已占用的端口号)
    return HCCL_SUCCESS;
}

HcclResult CollComm::GetHcclBinHandle(aclrtBinHandle& binHcclHandle)
{
    std::lock_guard<std::mutex> lock(binHcclmutex_);
    HCCL_DEBUG("[%s] GetHcclBinHandle", __func__);
    if (binHcclHandle_ == nullptr) {
        std::string hcclJsonPath;
        CHK_RET(GetKernelFilePath(hcclJsonPath));
        hcclJsonPath += "libscatter_aicpu_kernel.json";
        HcclResult ret
            = LoadBinaryFromFile(hcclJsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, binHcclHandle_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s]errNo[0x%016llx]load aicpu file fail, path[%s] optionType[%u] cpuKernelMode[%u].", __func__,
                HCCL_ERROR_CODE(ret), hcclJsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0),
            ret);

        HCCL_INFO(
            "[%s]load aicpu file success, path[%s] optionType[%u] cpuKernelMode[%u].", __func__, hcclJsonPath.c_str(),
            ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0);
    }
    binHcclHandle = binHcclHandle_;
    return HCCL_SUCCESS;
}

HcclResult CollComm::HcclBinaryUnLoad()
{
    std::lock_guard<std::mutex> lock(binHcclmutex_);
    if (binHcclHandle_ == nullptr) {
        HCCL_RUN_WARNING("[%s] binHcclHandle is nullptr", __func__);
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[%s]aclrtBinaryUnLoad binHcclHandle", __func__);
    aclError ret = aclrtBinaryUnLoad(binHcclHandle_);
    binHcclHandle_ = nullptr;
    if (ret != 0) {
        HCCL_RUN_WARNING("[%s]aclrtBinaryUnLoad failed, aclRet[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
