/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_info.h"
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"

namespace Hccl {

using namespace std;

void TopoInfo::Deserialize(const nlohmann::json& topoInfoJson)
{
    std::string msgVersion = "[TopoInfo::Deserialize] error occurs when parser object of propName \"version\"";
    std::string msgPc = "[TopoInfo::Deserialize] error occurs when parser object of propName \"peer_count\"";
    std::string msgEc = "[TopoInfo::Deserialize] error occurs when parser object of propName \"edge_count\"";
    TRY_CATCH_THROW(InvalidParamsException, msgVersion, version = GetJsonProperty(topoInfoJson, "version"););
    TRY_CATCH_THROW(InvalidParamsException, msgPc, peerCount = GetJsonPropertyUInt(topoInfoJson, "peer_count"););
    TRY_CATCH_THROW(InvalidParamsException, msgEc, edgeCount = GetJsonPropertyUInt(topoInfoJson, "edge_count"););

    if (version != "2.0") {
        HCCL_ERROR("[TopoInfo::%s] failed with version[%s] is not \"2.0\".", __func__, version.c_str());
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] failed with version[%s] is not \"2.0\" in topo file.", __func__, version.c_str()));
    }

    if (peerCount == 0 || peerCount > MAX_PEER_COUNT) {
        THROW<InvalidParamsException>(
            "[TopoInfo::%s] the range for the prop peer_count is [1, %u] while peer_count is %u", __func__,
            MAX_PEER_COUNT, peerCount);
    }

    if (edgeCount == 0) {
        HCCL_WARNING("[TopoInfo::%s]: edge_count is zero", __func__);
    }

    if (peerCount == 0 || peerCount > MAX_PEER_COUNT) {
        THROW<InvalidParamsException>(
            "[TopoInfo::%s] the range for the prop peer_count is [1, 65] while peer_count is %u", __func__, peerCount);
    }

    DeserializePeers(topoInfoJson);
    DeserializeEdges(topoInfoJson);

    u32 peer2PeerEdgeCount = 0;
    u32 peer2NetEdgeCount = 0;
    for (const auto& edge : edges) {
        if (edge.linkType == LinkType::PEER2PEER) {
            ++peer2PeerEdgeCount;
        } else if (edge.linkType == LinkType::PEER2NET) {
            ++peer2NetEdgeCount;
        }
    }

    HCCL_DEBUG(
        "[TopoInfo::%s] deserialize summary: peers[%zu], edges[%zu], PEER2PEER[%u], PEER2NET[%u]", __func__,
        peers.size(), edges.size(), peer2PeerEdgeCount, peer2NetEdgeCount);
}

void TopoInfo::DeserializePeers(const nlohmann::json& topoInfoJson)
{
    nlohmann::json peerJsons;
    std::string msgPl = "[TopoInfo::DeserializePeers] error occurs when parser object of propName \"peer_list\"";
    TRY_CATCH_THROW(InvalidParamsException, msgPl, GetJsonPropertyList(topoInfoJson, "peer_list", peerJsons););
    for (auto& peerJson : peerJsons) {
        PeerInfo peer;
        peer.Deserialize(peerJson);

        if (idSet.count(peer.localId) > 0) {
            THROW<InvalidParamsException>(
                StringFormat("[TopoInfo::%s] in peers exist duplicate localId = %u.", __func__, peer.localId));
        }

        peers.emplace_back(peer);
        idSet.insert(peer.localId);
    }

    if (peerCount != peers.size()) {
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] Value of peer_count[%u] is inconsistent with the size of the peer_list[%zu].", __func__,
            peerCount, peers.size()));
    }
}

void TopoInfo::VerifyEdges(EdgeInfo& edge)
{
    if (idSet.count(edge.localA) == 0 || idSet.count(edge.localB) == 0) {
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] endpoint localId [%u] or [%u] is not exist in peers[%zu].", __func__, edge.localA,
            edge.localB, idSet.size()));
    }
    // net_layer 不参与边语义；线性查重合并旧 topo 重复边，复杂度 O(E²)。
    if (find(edges.begin(), edges.end(), edge) != edges.end()) {
        HCCL_DEBUG(
            "[TopoInfo::%s] ignore duplicate physical edge, linkType[%s], topoType[%s], "
            "topoInstanceId[%u], localA[%u], localB[%u].",
            __func__, edge.linkType.Describe().c_str(), edge.topoType.Describe().c_str(), edge.topoInstId, edge.localA,
            edge.localB);
        return;
    }

    edges.emplace_back(edge);
}

void TopoInfo::DeserializeEdges(const nlohmann::json& topoInfoJson)
{
    nlohmann::json edgeJsons;
    std::string msgPe = "[TopoInfo::DeserializeEdges] error occurs when parser object of propName \"edge_list\"";
    TRY_CATCH_THROW(InvalidParamsException, msgPe, GetJsonPropertyList(topoInfoJson, "edge_list", edgeJsons););
    if (edgeJsons.empty()) {
        if (edgeCount != 0) {
            THROW<InvalidParamsException>(StringFormat(
                "[TopoInfo::%s] Value of edge_count[%u] is inconsistent with the size of edge_list[0].", __func__,
                edgeCount));
        } else {
            HCCL_WARNING("[TopoInfo::%s] edge count is zero", __func__);
            return;
        }
    }
    if (edgeJsons.size() != edgeCount) {
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] Value of edge_count[%u] is inconsistent with the size of edge_list[%zu].", __func__,
            edgeCount, edgeJsons.size()));
    }
    for (auto& edgeJson : edgeJsons) {
        EdgeInfo edge;
        edge.Deserialize(edgeJson);
        VerifyEdges(edge);
    }

    // edgeCount 对外表示去重后的物理边数量。
    edgeCount = static_cast<u32>(edges.size());
}

string TopoInfo::Describe() const
{
    string description = "TopoInfo{";
    description += StringFormat("version=%s", version.c_str());
    description += StringFormat(", peer_count=%u", peerCount);
    description += StringFormat(", edge_count=%u", edgeCount);
    description += StringFormat(", Peers.size=%u", peers.size());
    description += StringFormat(", Edges.size=%u", edges.size());
    description += "}";
    return description;
}

void TopoInfo::Dump() const
{
    HCCL_DEBUG("TopoInfo Dump:");
    HCCL_DEBUG("%s", Describe().c_str());
    HCCL_DEBUG("peers:");
    for (const auto& peer : peers) {
        HCCL_DEBUG("%s", peer.Describe().c_str());
    }
    HCCL_DEBUG("edges:");
    for (const auto& edge : edges) {
        HCCL_DEBUG("    %s", edge.Describe().c_str());
    }
}

TopoInfo::TopoInfo(BinaryStream& binaryStream)
{
    binaryStream >> version >> peerCount >> edgeCount;
    size_t peersSize = 0;
    binaryStream >> peersSize;
    for (u32 i = 0; i < peersSize; i++) {
        PeerInfo peer(binaryStream);
        peers.emplace_back(peer);
    }
    size_t edgeGroupCount = 0;
    binaryStream >> edgeGroupCount;

    HCCL_INFO(
        "[TopoInfo] version is [%s], peerCount is [%u], edgeCount is [%u], peers size is [%zu], "
        "edge group count is [%zu]",
        version.c_str(), peerCount, edgeCount, peers.size(), edgeGroupCount);
    // 保留旧快照分组外壳，读取后统一展开物理边。
    for (size_t i = 0; i < edgeGroupCount; i++) {
        u32 ignoredBinaryLayer = 0;
        size_t groupEdgeCount = 0;
        binaryStream >> ignoredBinaryLayer >> groupEdgeCount;
        for (size_t j = 0; j < groupEdgeCount; j++) {
            EdgeInfo edge(binaryStream);
            // 旧快照可能按 layer 重复存储同一物理边，展开时合并。
            if (find(edges.begin(), edges.end(), edge) == edges.end()) {
                edges.emplace_back(edge);
            }
        }
    }
    edgeCount = static_cast<u32>(edges.size());
}

void TopoInfo::GetBinStream(BinaryStream& binaryStream) const
{
    binaryStream << version << peerCount << edgeCount;
    binaryStream << peers.size();
    for (auto& it : peers) {
        it.GetBinStream(binaryStream);
    }
    const size_t edgeGroupCount = edges.empty() ? 0 : 1;
    binaryStream << edgeGroupCount;
    HCCL_INFO(
        "[TopoInfo::GetBinStream] version is [%s], peerCount is [%u], edgeCount is [%u], peers size is [%zu], "
        "edges size is [%zu]",
        version.c_str(), peerCount, edgeCount, peers.size(), edges.size());
    // 新快照统一写入一个兼容分组，不恢复逻辑分层。
    if (!edges.empty()) {
        binaryStream << LEGACY_BINARY_LAYER << edges.size();
        for (const auto& edge : edges) {
            edge.GetBinStream(binaryStream);
        }
    }
}

} // namespace Hccl
