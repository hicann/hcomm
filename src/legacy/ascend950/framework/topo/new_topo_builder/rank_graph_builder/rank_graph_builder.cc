/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <iterator>
#include "rank_graph_builder.h"
#include "detour_service.h"
#include "json_parser.h"
#include "phy_topo_builder.h"

namespace Hccl {

using namespace std;

unique_ptr<RankGraph> RankGraphBuilder::Build(const string &ranktableM, const string &topoPath, RankId myRank)
{
    PhyTopoBuilder::GetInstance().Build(topoPath);
    topoInfo_ = PhyTopoBuilder::GetInstance().GetTopoInfo();

    JsonParser    rankTableParser;
    RankTableInfo rankTableInfo;
    rankTableParser.ParseString(ranktableM, rankTableInfo);
    rankTable_ = make_unique<RankTableInfo>(rankTableInfo);

    this->myRank_ = myRank;
    BuildRankGraph();

    HCCL_INFO("[RankGraphBuilder] Build VirtualTopo success!");
    rankGraph_->Dump();
    return std::move(rankGraph_);
}

unique_ptr<RankGraph> RankGraphBuilder::Build(const RankTableInfo &ranktable, const string &topoPath, RankId myRank)
{
    PhyTopoBuilder::GetInstance().Build(topoPath);
    topoInfo_  = PhyTopoBuilder::GetInstance().GetTopoInfo();
    rankTable_ = make_unique<RankTableInfo>(ranktable);

    myRank_ = myRank;
    BuildRankGraph();

    HCCL_INFO("[RankGraphBuilder] Build VirtualTopo success!");
    rankGraph_->Dump();
    return std::move(rankGraph_);
}

const RankLevelInfo &RankGraphBuilder::GetRankLevelInfoByNetLayer(const NewRankInfo &rankInfo, u32 netLayer) const
{
    auto it = std::find_if(rankInfo.rankLevelInfos.begin(), rankInfo.rankLevelInfos.end(),
        [netLayer](const RankLevelInfo &levelInfo) {
            return levelInfo.netLayer == netLayer;
        });
    if (it == rankInfo.rankLevelInfos.end()) {
        THROW<InvalidParamsException>(StringFormat(
            "[RankGraphBuilder][GetRankLevelInfoByNetLayer] rankId[%u] netLayer[%u] does not exist in ranktable.",
            rankInfo.rankId, netLayer));
    }
    return *it;
}

u32 RankGraphBuilder::GetLocalDeviceId() const
{
    if (rankGraph_ == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][GetLocalDeviceId] rankGraph is nullptr"));
    }
    auto peer = rankGraph_->GetPeer(myRank_);
    if (peer == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][GetLocalDeviceId] local peer is nullptr"));
    }
    return peer->GetDeviceId();
}

std::vector<shared_ptr<PhyTopo::Link>> GetPeer2NetPhyLinks(LocalId localId)
{
    const shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> phyGraph = PhyTopo::GetInstance()->GetTopoGraph();
    if (phyGraph == nullptr) {
        THROW<InvalidParamsException>(
            StringFormat("[RankGraphBuilder][GetPhyLink] physical topo graph is null for localId[%d].", localId));
    }
    std::vector<shared_ptr<PhyTopo::Link>> links;
    // 统一物理图包含多种边，此处仅收集 PEER2NET 边。
    phyGraph->TraverseEdge(PhyTopo::Peer::GetId(localId), [&](shared_ptr<PhyTopo::Link> link) {
        if (link != nullptr && link->GetType() == LinkType::PEER2NET) {
            links.push_back(link);
        }
    });

    if (links.empty()) {
        THROW<InvalidParamsException>(
            StringFormat("[RankGraphBuilder][GetPhyLink] SourceNode localId[%d] edge does not exist.", localId));
    }
    return links;
}

bool IsPeer2NetLinkMatched(const shared_ptr<PhyTopo::Link> &link, const AddressInfo &addrInfo)
{
    if (link == nullptr || link->GetType() != LinkType::PEER2NET || link->GetSourceIFace() == nullptr) {
        return false;
    }
    // 端口有交集时，该物理边才属于当前 RankTable 地址。
    const auto &phyPorts = link->GetSourceIFace()->GetPorts();
    return std::any_of(addrInfo.ports.begin(), addrInfo.ports.end(),
        [&phyPorts](const std::string &port) { return phyPorts.count(port) != 0; });
}

const vector<shared_ptr<PhyTopo::Link>> &RankGraphBuilder::GetPeer2NetPhyLinksCached(LocalId localId)
{
    auto iter = peer2NetPhyLinksCache_.find(localId);
    if (iter == peer2NetPhyLinksCache_.end()) {
        iter = peer2NetPhyLinksCache_.emplace(localId, GetPeer2NetPhyLinks(localId)).first;
    }
    return iter->second;
}

void RankGraphBuilder::AddPeer2NetLink(const u32 netLayer,  const string &netInstId, RankId rankId, const AddressInfo &addrInfo,
                                      const shared_ptr<NetInstance::Fabric> &fabNode,
                                      const vector<shared_ptr<PhyTopo::Link>> &matchedLinks)
{
    for (shared_ptr<PhyTopo::Link> link : matchedLinks) {
        if (link == nullptr || link->GetSourceIFace() == nullptr) {
            continue;
        }
        std::set<std::string> ports = link->GetSourceIFace()->GetPorts();
        std::set<std::string> rankGraphPorts;
        std::set_intersection(ports.begin(), ports.end(), addrInfo.ports.begin(), addrInfo.ports.end(), 
            std::inserter(rankGraphPorts, rankGraphPorts.begin()));
        
        if (rankGraphPorts.empty()) {
            // 该地址在topo里没有对应边
            continue;
        }
        // 获取topoInstId topoType
        u32 topoInstId = link->GetTopoInstId();
        auto  topoType = link->GetTopoType();

        // 构造 RankGraph 的 PeerIface
        shared_ptr<NetInstance::ConnInterface> peerIface = make_shared<NetInstance::ConnInterface>(
            addrInfo.addr, rankGraphPorts, link->GetSourceIFace()->GetPos(), LinkType::PEER2NET, link->GetLinkProtocols(), topoType, topoInstId);
        // 获取 rankId 对应 PeerNode
        shared_ptr<NetInstance::Peer> peerNode = peers_.at(rankId);
        peerNode->AddConnInterface(netLayer, peerIface);

        // 构造 peer2netLink 和 net2peerLink 两条link
        shared_ptr<NetInstance::Link> peer2netLink =
            make_shared<NetInstance::Link>(peerNode, fabNode, peerIface, nullptr, LinkType::PEER2NET,
                                           link->GetLinkProtocols(), LinkDirection::BOTH, 2);
        shared_ptr<NetInstance::Link> net2peerLink =
            make_shared<NetInstance::Link>(fabNode, peerNode, nullptr, peerIface, LinkType::PEER2NET,
                                           link->GetLinkProtocols(), LinkDirection::BOTH, 2);

        // 插入 link
        tempNetInsts_[netLayer][netInstId]->AddLink(peer2netLink);
        tempNetInsts_[netLayer][netInstId]->AddLink(net2peerLink);

        // 将rank插入到当前netInstance对应的topoInstance中
        tempNetInsts_[netLayer][netInstId]->UpdateTopoInst(topoInstId, topoType, rankId);

        // 只打印当前卡的rank_id和eid对应关系
        if (rankId == myRank_) {
            HCCL_RUN_INFO("[RankGraphBuilder][AddPeer2NetLink] Add Peer2NetLink Net2PeerLink success. level[%u] "
                       "netInstId[%s] rankId[%u] planeId[%s] AddrStr[%s],topoInstId[%u],topoType[%u]",
                netLayer,  netInstId.c_str(), rankId, fabNode->GetPlaneId().c_str(), addrInfo.addr.Describe().c_str(),
                topoInstId, topoType);
        }
    }
}

void RankGraphBuilder::AddFabricInfo(u32 netLayer)
{
    auto netInst = rankGraph_->GetNetInstanceByRankId(netLayer, myRank_);
    if (netInst == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][AddFabricInfo] rankGraph->GetNetInstanceByRankId is nullptr"));
    } 

    if (netInst->GetNetType() != NetType::CLOS) {
        THROW<NotSupportException>(StringFormat("[RankGraphBuilder][AddFabricInfo] NetInstance is not CLOS, not support add fabric."));
    }
    string      netInstId = netInst->GetNetInstId();
    const auto &myLevelInfo = GetRankLevelInfoByNetLayer(rankTable_->ranks[myRank_], netLayer);
    // 根据planeId确认Fabric个数，每个fabricId对应一个planeId
    std::map<PlaneId, FabricId> planeId2Node = GetFabricsFromAddrInfo(myLevelInfo.rankAddrs);

    if (planeId2Node.size() == 0) {
        HCCL_WARNING("[RankGraphBuilder][AddFabricInfo] current rankId[%d] netLayer[%u] group no net plane", myRank_, netLayer);
        return;
    }
    vector<shared_ptr<NetInstance::Fabric>> fabNodes(planeId2Node.size(), nullptr);

    // 遍历每一个rankId，每个rankId都增加 peer2net 和 net2peer 两条链路
    for (RankId srcRankId : netInst->GetRankIds()) {
        const auto &srcLevelInfo = GetRankLevelInfoByNetLayer(rankTable_->ranks[srcRankId], netLayer);
        // rankId对应的物理逻辑localId
        LocalId localId  = rankGraph_->GetLocalId(srcRankId);
        // 从物理拓扑图中找出 localId 的所有 peer2Net 边。
        const auto &links = GetPeer2NetPhyLinksCached(localId);
        // 遍历ranktable中的addr，有几个addr就有几条peer2net的边
        for (const AddressInfo &addrInfo : srcLevelInfo.rankAddrs) {
            if (addrInfo.addr == IpAddress() || planeId2Node.count(addrInfo.planeId) == 0) {
                continue;
            }
            vector<shared_ptr<PhyTopo::Link>> matchedLinks;
            matchedLinks.reserve(links.size());
            std::copy_if(links.begin(), links.end(), std::back_inserter(matchedLinks),
                [&addrInfo](const shared_ptr<PhyTopo::Link> &link) {
                    return IsPeer2NetLinkMatched(link, addrInfo);
                });
            if (matchedLinks.empty()) {
                continue;
            }
            FabricId fabId = planeId2Node[addrInfo.planeId];
            // 若 fabNodes[fabId] 不存在则创建 如果存在则获取fabNode
            shared_ptr<NetInstance::Fabric> fabNode;
            if (fabNodes[fabId] == nullptr) {
                fabNode = make_shared<NetInstance::Fabric>(fabId, addrInfo.planeId);
                tempNetInsts_[netLayer][netInstId]->AddNode(fabNode);
                fabNodes[fabId] = fabNode;
            } else {
                fabNode = fabNodes[fabId];
            }
            // 插入peer和fabric的peer2net和net2peer两条link
            AddPeer2NetLink(netLayer, netInstId, srcRankId, addrInfo, fabNode, matchedLinks);
        }
    }

    HCCL_DEBUG("[RankGraphBuilder][AddFabricInfo] netLayer [%u] netInstId[%s] Add Fabric Info success!", netLayer,
               netInstId.c_str());
}

void RankGraphBuilder::AddTopoDescFabricInfo()
{
    // 1. 获取物理拓扑图
    auto phyTopoGraph = PhyTopo::GetInstance()->GetTopoGraph();
    if (phyTopoGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][AddTopoDescFabricInfo] phyTopoGraph is nullptr"));
    }
    HCCL_INFO("[RankGraphBuilder][AddTopoDescFabricInfo] Successfully retrieved phyTopoGraph");

    // 2. 获取当前 NetInstance
    NetInstance* innerNetInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (innerNetInstance == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraphBuilder][AddTopoDescFabricInfo] rankGraph->GetNetInstanceByRankId is nullptr"));
    }
    std::string netInstId = innerNetInstance->GetNetInstId();
    std::set<RankId> rankIds = innerNetInstance->GetRankIds();

    // Fabric 的归属以 RankTable planeId 为准，topoInstId 仅保留为拓扑实例属性。
    const auto &myLevelInfo = GetRankLevelInfoByNetLayer(rankTable_->ranks[myRank_], 0);
    const std::map<PlaneId, FabricId> planeId2Node = GetFabricsFromAddrInfo(myLevelInfo.rankAddrs);
    if (planeId2Node.empty()) {
        HCCL_WARNING("[RankGraphBuilder][AddTopoDescFabricInfo] rankId[%u] layer0 has no plane", myRank_);
        return;
    }
    vector<shared_ptr<NetInstance::Fabric>> fabNodes(planeId2Node.size(), nullptr);

    // 3. 遍历所有 rank 节点，根据端口匹配到的 planeId 创建 Fabric。
    for (RankId rankId : rankIds) {
        LocalId localId = rankGraph_->GetLocalId(rankId);
        const auto &levelInfo = GetRankLevelInfoByNetLayer(rankTable_->ranks[rankId], 0);
        auto peer2netEdges = phyTopoGraph->GetEdges(localId, PhyTopo::Fabric::GetId());

        HCCL_RUN_INFO(
            "[RankGraphBuilder][AddTopoDescFabricInfo] Processing rank %d (localId: %u), found %zu peer2net edges",
            rankId, localId, peer2netEdges.size());

        for (const auto& link : peer2netEdges) {
            if (link == nullptr || link->GetType() != LinkType::PEER2NET || link->GetSourceIFace() == nullptr) {
                continue;
            }
            for (const auto &addrInfo : levelInfo.rankAddrs) {
                auto planeIter = planeId2Node.find(addrInfo.planeId);
                if (addrInfo.addr == IpAddress() || planeIter == planeId2Node.end() ||
                    !IsPeer2NetLinkMatched(link, addrInfo)) {
                    continue;
                }

                FabricId fabId = planeIter->second;
                shared_ptr<NetInstance::Fabric> fabNode = fabNodes[fabId];
                if (fabNode == nullptr) {
                    fabNode = make_shared<NetInstance::Fabric>(fabId, addrInfo.planeId);
                    innerNetInstance->AddNode(fabNode);
                    fabNodes[fabId] = fabNode;
                    HCCL_INFO("[RankGraphBuilder][AddTopoDescFabricInfo] create Fabric for planeId[%s], "
                              "fabricId[%u]",
                        addrInfo.planeId.c_str(), fabId);
                }

                const vector<shared_ptr<PhyTopo::Link>> matchedLinks = {link};
                AddPeer2NetLink(0, netInstId, rankId, addrInfo, fabNode, matchedLinks);
            }
        }
    }
    HCCL_INFO("[RankGraphBuilder][AddTopoDescFabricInfo] Successfully completed fabric link construction");
}

std::map<PlaneId, FabricId> GetFabricsFromAddrInfo(const std::vector<AddressInfo>& rankAddrs)
{
    std::map<PlaneId, FabricId> planeId2FabricId;
    for (const auto& addrInfo : rankAddrs) {
        if (planeId2FabricId.count(addrInfo.planeId) == 0) {
            FabricId fabId = planeId2FabricId.size();
            planeId2FabricId[addrInfo.planeId] = fabId;
        }
    }
    return planeId2FabricId;
}

// 根据ranktable构造添加peers和NetInstances, NetInstance添加nodes和links(peer2net)
// 1. 创建NetInstance ( 每个NetInstance 添加 Rank， Node， Link)；
// 2. RankGraph中添加NetInstance， Peer， Fabric，
void RankGraphBuilder::BuildFromRankTable()
{
    peer2NetPhyLinksCache_.clear();
    // 保存NetInstance指针以便后续执行Add操作
    tempNetInsts_.resize(MAX_NET_LAYER);   //为了方便修改RankGraph的NetInstance，共享指针。

    // 遍历rankTable每一个rank, virtualTopo添加Peers
    for (const auto &rankInfo : rankTable_->ranks) {
        updaterFor64Plus1_.SaveReplaceInfo(rankInfo);   // 暂存备份替换信息
        RankId rankId = rankInfo.rankId;
        shared_ptr<NetInstance::Peer> peer = make_shared<NetInstance::Peer>(rankId, rankInfo.localId, rankInfo.replacedLocalId, rankInfo.deviceId, rankInfo.devicePort, rankInfo.hostPort);
        rankGraph_->AddPeer(peer);
        peers_.emplace(rankId, peer);  // rankid2peer

        // 构造当前rank的每个LevelInfo所在NetInstance, 添加 RankId 和 Peer
        for (const auto &levelInfo : rankInfo.rankLevelInfos) {
            // rankLevelInfo.level、id对应NetInstance，若不存在则创建
            auto curNetInstance = GetOrCreateNetInstance(levelInfo.netLayer, levelInfo.netInstId, levelInfo.netType, tempNetInsts_, rankGraph_.get());
            if (curNetInstance == nullptr) {
                continue;
            }
            // NetInstance add Peer
            curNetInstance->AddRankId(rankId);
            curNetInstance->AddNode(peer);
            // Peer add NetInstance
            peer->AddNetInstance(curNetInstance);
            if (levelInfo.netLayer == 0) {
                peer->SetPortPortAddrMapLayer0(levelInfo.portAddrMap);
            }
            HCCL_DEBUG("[RankGraphBuilder][BuildFromRankTable] rankLevelInfo : rankId[%d] level[%u] "
                       "netInstId[%s] fabricType[%s].",
                       rankId, levelInfo.netLayer, levelInfo.netInstId.c_str(),
                       levelInfo.netType.Describe().c_str());
        }
    }

    // 对 myrank 所在每个level的NetInstance 添加 Fabrics 和 links(peer2net)
    set<u32> myLevels = rankGraph_->GetLevels(myRank_);
    HCCL_DEBUG("myRank netType: level size %u", myLevels.size());
    for (u32 level : myLevels) {
        if (level == 0) {
            AddTopoDescFabricInfo();
        } else {
            AddFabricInfo(level);
        }
    }

    // 初始化innerRanks
    rankGraph_->InitInnerRanks();

    HCCL_DEBUG("[RankGraphBuilder][BuildFromRankTable] Build VirtualTopo from RankTable success!");
}

void RankGraphBuilder::SetEndpointDesc()
{
    std::shared_ptr<NetInstance::Peer> peer = peers_[myRank_];
    CHK_PRT_THROW(peer == nullptr, HCCL_ERROR("[RankGraphBuilder::%s] fail", __func__), NullPtrException, "peer is null" );
    // 获取 peer 的 Iface
    std::set<u32> layers = peer->GetLevels();
    for (const auto& layer : layers) {
        auto ifacesVec = peer->GetIfacesByLayer(layer);
        for (const auto& iface : ifacesVec) {
            const auto& protocols = iface->GetLinkProtocols();
            for (const auto& protocol : protocols) {
                EndpointDesc desc{};

                HcclResult ret = GetCommAddr(desc.commAddr, iface->GetAddr());
                CHK_PRT_THROW(ret != HCCL_SUCCESS, HCCL_ERROR("[RankGraphBuilder::%s] fail", __func__), InternalException, "GetCommAddr fail" );

                desc.protocol = LinkProtocolToCommProtocol(protocol);
                desc.loc.locType = AddrPositionToEndpointLoc(iface->GetPos());

                HCCL_INFO("[RankGraphBuilder::SetEndpointDesc] local type[%d] protocol[%d]",
                          desc.loc.locType, desc.protocol);

                peer->SetEndpointToIface(desc.commAddr, desc.protocol, iface);
            }
        }
    }
}

std::shared_ptr<NetInstance> RankGraphBuilder::GetNetInstance(const RankLevelInfo &levelInfo){
    auto it = tempNetInsts_[levelInfo.netLayer].find(levelInfo.netInstId);
    if (it == tempNetInsts_[levelInfo.netLayer].end()) {
        return nullptr;
    }
    // 若NetInstance存在, type不一致则报错
    NetType netType = it->second->GetNetType();
    if (netType != levelInfo.netType) {
        HCCL_WARNING("[CreateNetInstance]FabType [%s] and [%s] no match", netType.Describe().c_str(),
                        levelInfo.netType.Describe().c_str());
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<NetInstance> RankGraphBuilder::CreateNetInstance(const RankLevelInfo &levelInfo)
{
    std::shared_ptr<NetInstance> netInst;
    if (levelInfo.netType == NetType::TOPO_FILE_DESC) {
        netInst = std::make_shared<InnerNetInstance>(levelInfo.netLayer, levelInfo.netInstId);
    } else if (levelInfo.netType == NetType::CLOS) {
        netInst = std::make_shared<ClosNetInstance>(levelInfo.netLayer, levelInfo.netInstId);
    } else {
        THROW<NotSupportException>(StringFormat("[RankGraphBuilder][CreateNetInstance] netType: %s is not support", levelInfo.netType));
    }
    return netInst;
}

// 从phytopo和ranktable中读取数据共同构建peer2peer的边。
void RankGraphBuilder::BuildPeer2PeerLinks()
{
    auto phyTopoGraph = PhyTopo::GetInstance()->GetTopoGraph();
    if (phyTopoGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][BuildPeer2PeerLinks] phyTopoGraph is nullptr"));
    }
    // 遍历innerNetInstance中的每两个rankId之间是否存在边，存在则添加peer2peerlink
    NetInstance *innerNetInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (innerNetInstance == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][BuildPeer2PeerLinks] innerNetInstance is nullptr"));
    }
    set<RankId> rankIds = innerNetInstance->GetRankIds();

    auto localDeviceId = GetLocalDeviceId();
    for (const auto srcRankId : rankIds) {
        for (const auto dstRankId : rankIds) {
           if (srcRankId == dstRankId) {
                continue;
           }

           // 得到phyTopoGraph中对应的localId
           LocalId srcLocalId = rankGraph_->GetLocalId(srcRankId);
           LocalId dstLocalId = rankGraph_->GetLocalId(dstRankId);
           if (srcLocalId == BACKUP_LOCAL_ID || dstLocalId == BACKUP_LOCAL_ID) {
                continue;
           }

           std::vector<shared_ptr<PhyTopo::Link>> phyLinks = GetPeer2PeerPhyLinks(phyTopoGraph, srcLocalId, dstLocalId);
           // 按 RankTable layer 0 端口筛选物理 P2P 边并补齐地址。

           shared_ptr<NetInstance::Peer> srcPeer = peers_.at(srcRankId);
           shared_ptr<NetInstance::Peer> dstPeer = peers_.at(dstRankId);

           for (shared_ptr<PhyTopo::Link> phyLink : phyLinks) {
                auto sourceIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
                    phyLink->GetSourceIFace(), srcPeer->GetPortAddrMapLayer0(), phyLink->GetTopoType(), phyLink->GetTopoInstId(), localDeviceId);
                auto targetIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
                    phyLink->GetTargetIFace(), dstPeer->GetPortAddrMapLayer0(), phyLink->GetTopoType(), phyLink->GetTopoInstId(), localDeviceId);
                if (sourceIfaces.empty() || targetIfaces.empty()) {
                    // 没有可用的接口。
                    HCCL_WARNING("[RankGraphBuilder][BuildPeer2PeerLinks] no available interface, "
                        "srcRankId[%d] dstRankId[%d].", srcRankId, dstRankId);
                    continue;
                }
                srcPeer->AddConnInterfaces(0, sourceIfaces);
                dstPeer->AddConnInterfaces(0, targetIfaces);
                std::vector<shared_ptr<NetInstance::Link>> links =
                    ConstructLinks(srcPeer, dstPeer, sourceIfaces, targetIfaces, phyLink);
                for (auto link : links) {
                    innerNetInstance->AddLink(link);
                }
           }
        }
    }
}

void RankGraphBuilder::UpdateTopoInstForMyRankOnly()
{
    auto innerNetInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (innerNetInstance == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][UpdateTopoInstForMyRankOnly] innerNetInstance is nullptr"));
    }

    auto netInstId = innerNetInstance->GetNetInstId();
    set<RankId> rankIds = innerNetInstance->GetRankIds();

    auto localDeviceId = GetLocalDeviceId();
    auto phyTopoGraph = PhyTopo::GetInstance()->GetTopoGraph();
    if (phyTopoGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][UpdateTopoInstForMyRankOnly] phyTopoGraph is nullptr"));
    }
    if (rankIds.size() == 1) {
        // 单卡场景直接返回1DMESH
        RankId singleId = *rankIds.begin();
        tempNetInsts_[0][netInstId]->UpdateTopoInst(0, TopoType::MESH_1D, singleId);
        return;
    }

    for (const auto srcRankId : rankIds) {
        for (const auto dstRankId : rankIds) {
            // 只处理涉及 myRank_ 的边
            if (srcRankId == dstRankId || (srcRankId != myRank_ && dstRankId != myRank_)) {
                continue;
            }

            LocalId srcLocalId = rankGraph_->GetLocalId(srcRankId);
            LocalId dstLocalId = rankGraph_->GetLocalId(dstRankId);
            if (srcLocalId == BACKUP_LOCAL_ID || dstLocalId == BACKUP_LOCAL_ID) {
                continue;
            }

            std::vector<shared_ptr<PhyTopo::Link>> phyLinks =
                GetPeer2PeerPhyLinks(phyTopoGraph, srcLocalId, dstLocalId);
            // 通过 RankTable layer 0 端口映射物理 P2P 链路。
            const auto &srcLevelInfo = GetRankLevelInfoByNetLayer(rankTable_->ranks[srcRankId], 0);
            const auto &dstLevelInfo = GetRankLevelInfoByNetLayer(rankTable_->ranks[dstRankId], 0);

            for (shared_ptr<PhyTopo::Link> phyLink : phyLinks) {
                auto sourceIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
                    phyLink->GetSourceIFace(), srcLevelInfo.portAddrMap, phyLink->GetTopoType(),
                    phyLink->GetTopoInstId(), localDeviceId);
                auto targetIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
                    phyLink->GetTargetIFace(), dstLevelInfo.portAddrMap, phyLink->GetTopoType(),
                    phyLink->GetTopoInstId(), localDeviceId);
                if (sourceIfaces.empty() || targetIfaces.empty()) {
                    continue;
                }
                tempNetInsts_[0][netInstId]->UpdateTopoInst(
                    phyLink->GetTopoInstId(), phyLink->GetTopoType(), dstRankId);
            }
        }
    }
}

std::vector<std::shared_ptr<NetInstance::ConnInterface>> ConstructConnIFromPhyTopoConnIAndPortMap(
        std::shared_ptr<PhyTopo::ConnInterface> phyConnIFace, const std::map<std::string, std::vector<IpAddress>>& portAddrMap, 
        const TopoType topoType, const u32 topoInstId, u32 localDeviceId) {
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> netConnIFaces;
    std::set<string> phyPorts = phyConnIFace->GetPorts();
    std::map<IpAddress, std::set<string>> addr2Ports;
    // 非 PCIe 端口仅保留 RankTable 中存在的物理端口。
    for (auto port: phyPorts) {
        if (*(phyConnIFace->GetLinkProtocols().begin()) == LinkProtocol::PCIE) {
            IpAddress tempIp;
            HrtRaSocketGetVnicIpInfos(localDeviceId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, localDeviceId, tempIp);
            auto it = addr2Ports.find(tempIp);
            if (it == addr2Ports.end()) {
                std::set<std::string> newPorts;
                newPorts.insert("d2h");
                addr2Ports[tempIp] = newPorts;
            } else {
                it->second.insert("d2h");
            }
        } else {
            auto itPort = portAddrMap.find(port);
            if (itPort == portAddrMap.end()) {
                HCCL_WARNING("[RankGraphBuilder][ConstructConnIFromPhyTopoConnIAndPortMap] topo use port [%s] not find addrs in ranktable.", port.c_str());
                continue;
            }
            for (auto addr : itPort->second) {
                auto it = addr2Ports.find(addr);
                if (it == addr2Ports.end()) {
                    std::set<std::string> newPorts;
                    newPorts.insert(port);
                    addr2Ports[addr] = newPorts;
                } else {
                    it->second.insert("8080");
                }
            }
        }
    }

    for (auto it = addr2Ports.begin(); it != addr2Ports.end(); ++it) {
        auto linkType = *(phyConnIFace->GetLinkProtocols().begin()) == LinkProtocol::PCIE ? LinkType::PEER2NET : LinkType::PEER2PEER;
        shared_ptr<NetInstance::ConnInterface> netConnIFace =
            make_shared<NetInstance::ConnInterface>(it->first, it->second, phyConnIFace->GetPos(), linkType,
                                                    phyConnIFace->GetLinkProtocols(), topoType, topoInstId);
        netConnIFaces.push_back(netConnIFace);
    }
    return netConnIFaces;
}

std::vector<shared_ptr<NetInstance::Link>> ConstructLinks(shared_ptr<NetInstance::Peer> srcPeer, shared_ptr<NetInstance::Peer> dstPeer,
        std::vector<std::shared_ptr<NetInstance::ConnInterface>> sourceIfaces,
        std::vector<std::shared_ptr<NetInstance::ConnInterface>> targetIfaces, shared_ptr<PhyTopo::Link> phyLink) 
{
    std::vector<shared_ptr<NetInstance::Link>> links;
    for (auto sourceIFace : sourceIfaces) {
        for (auto targetIFace : targetIfaces) {
            shared_ptr<NetInstance::Link> link = make_shared<NetInstance::Link>(srcPeer, dstPeer, sourceIFace, targetIFace,
                                                                          LinkType::PEER2PEER, phyLink->GetLinkProtocols());
            links.push_back(link);
        }
    }
    return links;
}

std::vector<std::shared_ptr<PhyTopo::Link>> GetPeer2PeerPhyLinks(std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> phyTopoGraph, LocalId srcLocalId, LocalId dstLocalId)
{
    std::vector<shared_ptr<PhyTopo::Link>> links;
    if (!phyTopoGraph->HasNode(srcLocalId) || !phyTopoGraph->HasNode(dstLocalId)) {
        HCCL_WARNING("[RankGraphBuilder][BuildFromPhytopo] srcLocalId[%u] dstLocalId[%u] not exist in phyTopoGraph.",
            srcLocalId,
            dstLocalId);
        return links;
    }
    // 得到phyTopoGraph对应的NodeId
    NodeId srcNodeId = PhyTopo::Peer::GetId(srcLocalId);
    NodeId dstNodeId = PhyTopo::Peer::GetId(dstLocalId);

    phyTopoGraph->TraverseEdge(srcNodeId, dstNodeId, [&](shared_ptr<PhyTopo::Link> link) {
        if (link != nullptr && link->GetType() == LinkType::PEER2PEER) {
            links.push_back(link);
        }
    });
    if (links.empty()) {
        HCCL_WARNING(
            "[RankGraphBuilder][GetPeer2PeerPhyLinks] srcLocalId[%u] dstLocalId[%u] edge does not exist.", srcLocalId, dstLocalId);
    }
    return links;
}

void RankGraphBuilder::CheckMyRankInRankTable() const
{
    if (myRank_ >= static_cast<s32>(rankTable_->rankCount)) {
        THROW<InvalidParamsException>(StringFormat("[RankGraphBuilder][CheckMyRankInRankTable]"
            "myRank[%d] is not in rankTable rankCount[%u].", myRank_, rankTable_->rankCount));
    }
}

void RankGraphBuilder::BuildRankGraph()
{
    // 创建VirtualTopo
    rankGraph_ = make_unique<RankGraph>(myRank_);

    // 校验myRank在rankTable中
    CheckMyRankInRankTable();

    // 根据ranktable构造添加peers和NetInstances, 每个NetInstance添加nodes和links(peer2net)
    BuildFromRankTable();

    // 根据phytopo构造添加InnerGroup中的links(peer2peer), 不包括备份节点
    BuildPeer2PeerLinks();

    // 使用备份D时需要修改虚拟拓扑
    updaterFor64Plus1_.UpdateRankGraph(rankGraph_.get(), rankTable_.get());

    // 为myrank的peer2peer更新topoInst
    UpdateTopoInstForMyRankOnly();

    // 添加绕路 绕路获取
    DetourService::GetInstance().InsertDetourLinks(rankGraph_.get(), rankTable_.get());

    // 设置endpoint
    SetEndpointDesc();

    // 构造完成
    rankGraph_->InitFinish();
}

std::unique_ptr<RankTableInfo> RankGraphBuilder::GetRankTableInfo()
{
    return move(rankTable_);
}

std::shared_ptr<TopoInfo> RankGraphBuilder::GetTopoInfo()
{
    return  topoInfo_;
}

unique_ptr<RankGraph> RankGraphBuilder::RecoverBuild(const RankTableInfo &rankTableInfo,const TopoInfo &topoInfo, RankId myRank)
{
    topoInfo_ = std::make_shared<TopoInfo>(topoInfo);
    PhyTopoBuilder::GetInstance().RecoverBuild(*topoInfo_);

    rankTable_ = make_unique<RankTableInfo>(rankTableInfo);
    HCCL_INFO("[%s] RankTable[%s] RankTableInfo[%s]", __func__, rankTable_->Describe().c_str(),
              rankTableInfo.Describe().c_str());

    this->myRank_ = myRank;
    BuildRankGraph();

    HCCL_INFO("[RankGraphBuilder] Build VirtualTopo success!");
    rankGraph_->Dump();
    return std::move(rankGraph_);
}

} // namespace Hccl
