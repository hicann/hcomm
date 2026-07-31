/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_GRAPH_BUILDER_H
#define RANK_GRAPH_BUILDER_H

#include <memory>
#include <map>

#include "json_parser.h"
#include "rank_gph.h"
#include "rank_table_info.h"
#include "phy_topo.h"
#include "types.h"
#include "topo_common_types.h"
#include "topo_info.h"
#include "updater_for_64_plus_1.h"

namespace Hccl {

class RankGraphBuilder {
public:
    std::unique_ptr<RankGraph> Build(const std::string &ranktableM, const std::string &topoPath, RankId myRank);
    unique_ptr<RankGraph>      Build(const RankTableInfo &ranktable, const string &topoPath, RankId myRank);
    std::unique_ptr<RankTableInfo> GetRankTableInfo();
    std::shared_ptr<TopoInfo>      GetTopoInfo();
    unique_ptr<RankGraph> RecoverBuild(const RankTableInfo &rankTableInfo,const TopoInfo &topoInfo, RankId myRank);
    void SetEndpointDesc();

private:
    std::unique_ptr<RankTableInfo>  rankTable_;
    std::unique_ptr<RankGraph>      rankGraph_;
    RankId2PeerMap                  peers_;
    Level2Id2NetInst                tempNetInsts_;
    RankId                          myRank_;
    std::shared_ptr<TopoInfo>       topoInfo_;
    UpdaterFor64Plus1               updaterFor64Plus1_{};

    void CheckMyRankInRankTable() const;
    void CheckNetLayerFromPhyTopo(const u32 netLayer) const;
    void BuildRankGraph();
    void BuildFromRankTable();
    void BuildPeer2PeerLinks();
    void AddFabricInfo(u32 level);
    void AddPeer2NetLink(const u32 netLayer,  const string &netInstId, RankId rankId, const AddressInfo &addrInfo,
                        const shared_ptr<NetInstance::Fabric> &fabNode, const vector<shared_ptr<PhyTopo::Link>> &links);
    void AddTopoDescFabricInfo();
    void UpdateTopoInstForMyRankOnly();
    const RankLevelInfo &GetRankLevelInfoByNetLayer(const NewRankInfo &rankInfo, u32 netLayer) const;
    // 新增创建NetInstance
    std::shared_ptr<NetInstance> GetNetInstance(const RankLevelInfo &levelInfo);
    std::shared_ptr<NetInstance> CreateNetInstance(const RankLevelInfo &levelInfo);
};

std::map<PlaneId, FabricId> GetFabricsFromAddrInfo(const std::vector<AddressInfo>& rankAddrs);
std::vector<shared_ptr<PhyTopo::Link>> GetPeer2NetPhyLinks(u32 netLayer, LocalId localId);
std::vector<std::shared_ptr<NetInstance::ConnInterface>> ConstructConnIFromPhyTopoConnIAndPortMap(
    std::shared_ptr<PhyTopo::ConnInterface> phyConnIFace, const std::map<std::string, std::vector<IpAddress>>& portAddrMap,
    const TopoType topoType, const u32 topoInstId, u32 localDeviceId);
std::vector<shared_ptr<NetInstance::Link>> ConstructLinks(shared_ptr<NetInstance::Peer> srcPeer,
    shared_ptr<NetInstance::Peer> dstPeer, std::vector<std::shared_ptr<NetInstance::ConnInterface>> sourceIfaces,
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> targetIfaces, shared_ptr<PhyTopo::Link> phyLink);
std::vector<std::shared_ptr<PhyTopo::Link>> GetPeer2PeerPhyLinks(std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> phyTopoGraph,
    LocalId srcLocalId, LocalId dstLocalId);
} // namespace Hccl

#endif // RANK_GRAPH_BUILDER_H
