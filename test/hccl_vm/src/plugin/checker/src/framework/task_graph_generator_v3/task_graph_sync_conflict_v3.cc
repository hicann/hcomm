/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "task_graph_sync_conflict_v3.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "sim_log.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
namespace {
enum class SyncResourceKind : uint8_t {
    AICPU_NOTIFY = 0,
    CCU_CKE,
};

struct SyncResourceKey {
    SyncResourceKind kind{SyncResourceKind::AICPU_NOTIFY};
    uint32_t notifyId{INVALID_NOTIFY_ID};
    RankId rankId{INVALID_RANK_ID};
    uint32_t dieId{INVALID_DIE_ID};
    uint16_t ckeId{INVALID_CCU_CKE};
    uint8_t ckeBit{0};

    bool operator<(const SyncResourceKey &rhs) const
    {
        return std::tie(kind, notifyId, rankId, dieId, ckeId, ckeBit) <
            std::tie(rhs.kind, rhs.notifyId, rhs.rankId, rhs.dieId, rhs.ckeId, rhs.ckeBit);
    }

    bool operator==(const SyncResourceKey &rhs) const
    {
        return !(*this < rhs) && !(rhs < *this);
    }
};

enum class CopyNodeRole : uint8_t {
    NORMAL = 0,
    START_WAIT,
    RECEIVE_WAIT,
};

struct CopyNode {
    const TaskNode *origin{nullptr};
    CopyNodeRole role{CopyNodeRole::NORMAL};
    size_t index{0};
    std::vector<CopyNode *> parents;
    std::vector<CopyNode *> children;
};

struct CopiedGraph {
    std::vector<std::unique_ptr<CopyNode>> nodes;
    std::map<const TaskNode *, CopyNode *> normalNodes;
    std::map<const TaskNode *, CopyNode *> startWaitNodes;
    std::map<const TaskNode *, CopyNode *> receiveWaitNodes;
};

struct SyncPair {
    SyncResourceKey resource;
    const TaskNode *post{nullptr};
    const TaskNode *wait{nullptr};
    CopyNode *postCopy{nullptr};
    CopyNode *startWaitCopy{nullptr};
    CopyNode *receiveWaitCopy{nullptr};
    size_t order{0};
};

struct SyncBucket {
    std::vector<SyncPair> pairs;
    std::set<const TaskNode *> originalPosts;
    std::set<const TaskNode *> originalWaits;
};

using SyncBuckets = std::map<SyncResourceKey, SyncBucket>;
using SyncParentMap = std::map<const TaskNode *, std::set<const TaskNode *>>;

const char *SyncResourceKindName(SyncResourceKind kind)
{
    switch (kind) {
        case SyncResourceKind::AICPU_NOTIFY:
            return "AICPU_NOTIFY";
        case SyncResourceKind::CCU_CKE:
            return "CCU_CKE";
        default:
            return "INVALID";
    }
}

const char *SyncPostTaskTypeName(SyncResourceKind kind)
{
    switch (kind) {
        case SyncResourceKind::AICPU_NOTIFY:
        case SyncResourceKind::CCU_CKE:
            return "RECORD";
        default:
            return "POST";
    }
}

const char *SyncWaitTaskTypeName(SyncResourceKind kind)
{
    switch (kind) {
        case SyncResourceKind::AICPU_NOTIFY:
        case SyncResourceKind::CCU_CKE:
            return "WAIT";
        default:
            return "WAIT";
    }
}

std::string DescribeResource(const SyncResourceKey &resource)
{
    std::ostringstream os;
    os << "kind=" << SyncResourceKindName(resource.kind);
    if (resource.kind == SyncResourceKind::AICPU_NOTIFY) {
        os << ", notifyId=" << resource.notifyId;
    } else if (resource.kind == SyncResourceKind::CCU_CKE) {
        os << ", waitRankId=" << resource.rankId << ", dieId=" << resource.dieId
           << ", ckeId=" << resource.ckeId << ", bit=" << static_cast<uint32_t>(resource.ckeBit);
    }
    return os.str();
}

bool IsWaitNode(const TaskNode *node)
{
    return node != nullptr && node->GetType() == TaskType::WAIT;
}

bool IsPostNode(const TaskNode *node)
{
    return node != nullptr && node->GetType() == TaskType::RECORD;
}

std::vector<SyncResourceKey> GetSyncResources(const TaskNode *node)
{
    std::vector<SyncResourceKey> resources;
    if (const auto *record = dynamic_cast<const TaskRecordAICPU *>(node)) {
        if (record->GetNotify().notifyId != INVALID_NOTIFY_ID) {
            resources.push_back(SyncResourceKey{SyncResourceKind::AICPU_NOTIFY,
                record->GetNotify().notifyId});
        }
        return resources;
    }
    if (const auto *wait = dynamic_cast<const TaskWaitAICPU *>(node)) {
        if (wait->GetNotify().notifyId != INVALID_NOTIFY_ID) {
            resources.push_back(SyncResourceKey{SyncResourceKind::AICPU_NOTIFY,
                wait->GetNotify().notifyId});
        }
        return resources;
    }
    if (const auto *record = dynamic_cast<const TaskRecordCCU *>(node)) {
        const CcuNotify &notify = record->GetNotify();
        if (notify.ckeId == INVALID_CCU_CKE) {
            return resources;
        }
        for (uint32_t bit = 0; bit < 16U; ++bit) {
            if ((notify.ckeMask & static_cast<uint16_t>(1U << bit)) != 0U) {
                resources.push_back(SyncResourceKey{SyncResourceKind::CCU_CKE, 0,
                    notify.waitRankId, notify.dieId, notify.ckeId, static_cast<uint8_t>(bit)});
            }
        }
        return resources;
    }
    if (const auto *wait = dynamic_cast<const TaskWaitCCU *>(node)) {
        const CcuNotify &notify = wait->GetNotify();
        if (notify.ckeId == INVALID_CCU_CKE) {
            return resources;
        }
        for (uint32_t bit = 0; bit < 16U; ++bit) {
            if ((notify.ckeMask & static_cast<uint16_t>(1U << bit)) != 0U) {
                resources.push_back(SyncResourceKey{SyncResourceKind::CCU_CKE, 0,
                    notify.waitRankId, notify.dieId, notify.ckeId, static_cast<uint8_t>(bit)});
            }
        }
    }
    return resources;
}

bool ContainsResource(const std::vector<SyncResourceKey> &resources, const SyncResourceKey &resource)
{
    return std::find(resources.begin(), resources.end(), resource) != resources.end();
}

HcclResult CollectOriginalNodes(const TaskNode *start, std::vector<const TaskNode *> &nodes)
{
    nodes.clear();
    if (start == nullptr) {
        return HCCL_E_PTR;
    }
    std::queue<const TaskNode *> pending;
    std::set<const TaskNode *> visited;
    pending.push(start);
    visited.insert(start);
    while (!pending.empty()) {
        const TaskNode *node = pending.front();
        pending.pop();
        nodes.push_back(node);
        for (const TaskNode *child : node->GetChildren()) {
            if (child == nullptr) {
                HCCL_VM_ERROR("Sync DAG is invalid because a child node is null, parent={}", node->Describe());
                return HCCL_E_PTR;
            }
            if (visited.insert(child).second) {
                pending.push(child);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BuildSyncPairs(const std::vector<const TaskNode *> &nodes, SyncBuckets &buckets,
    SyncParentMap &syncParents)
{
    buckets.clear();
    syncParents.clear();
    for (const TaskNode *node : nodes) {
        if (!IsPostNode(node)) {
            continue;
        }
        for (const SyncResourceKey &resource : GetSyncResources(node)) {
            buckets[resource].originalPosts.insert(node);
        }
    }
    for (const TaskNode *wait : nodes) {
        if (!IsWaitNode(wait)) {
            continue;
        }
        const std::vector<SyncResourceKey> waitResources = GetSyncResources(wait);
        for (const SyncResourceKey &resource : waitResources) {
            buckets[resource].originalWaits.insert(wait);
        }
        for (const TaskNode *parent : wait->GetParents()) {
            if (parent == nullptr) {
                HCCL_VM_ERROR("Sync DAG is invalid because a parent node is null, childWait={}", wait->Describe());
                return HCCL_E_PTR;
            }
            if (!IsPostNode(parent)) {
                continue;
            }
            for (const SyncResourceKey &resource : GetSyncResources(parent)) {
                if (ContainsResource(waitResources, resource)) {
                    syncParents[wait].insert(parent);
                    buckets[resource].pairs.push_back(SyncPair{resource, parent, wait});
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

CopyNode *FindCompletionCopy(const CopiedGraph &graph, const TaskNode *node)
{
    if (IsWaitNode(node)) {
        const auto iter = graph.receiveWaitNodes.find(node);
        return iter == graph.receiveWaitNodes.end() ? nullptr : iter->second;
    }
    const auto iter = graph.normalNodes.find(node);
    return iter == graph.normalNodes.end() ? nullptr : iter->second;
}

CopyNode *FindStartCopy(const CopiedGraph &graph, const TaskNode *node)
{
    if (IsWaitNode(node)) {
        const auto iter = graph.startWaitNodes.find(node);
        return iter == graph.startWaitNodes.end() ? nullptr : iter->second;
    }
    const auto iter = graph.normalNodes.find(node);
    return iter == graph.normalNodes.end() ? nullptr : iter->second;
}

void AddCopyEdge(CopyNode *parent, CopyNode *child)
{
    if (parent == nullptr || child == nullptr || parent == child) {
        return;
    }
    if (std::find(parent->children.begin(), parent->children.end(), child) == parent->children.end()) {
        parent->children.push_back(child);
    }
    if (std::find(child->parents.begin(), child->parents.end(), parent) == child->parents.end()) {
        child->parents.push_back(parent);
    }
}

bool IsSameRankStreamQueue(const TaskNode *lhs, const TaskNode *rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    const TaskPosition &lhsPosition = lhs->GetPosition();
    const TaskPosition &rhsPosition = rhs->GetPosition();
    return lhsPosition.rankId == rhsPosition.rankId && lhsPosition.streamId == rhsPosition.streamId &&
        lhsPosition.queueId == rhsPosition.queueId;
}

HcclResult BuildCopiedGraph(const std::vector<const TaskNode *> &nodes, const SyncParentMap &syncParents,
    CopiedGraph &graph)
{
    graph = CopiedGraph{};
    graph.nodes.reserve(nodes.size() + 1U);
    for (const TaskNode *origin : nodes) {
        if (origin == nullptr) {
            return HCCL_E_PTR;
        }
        if (IsWaitNode(origin)) {
            auto startWait = std::make_unique<CopyNode>();
            startWait->origin = origin;
            startWait->role = CopyNodeRole::START_WAIT;
            startWait->index = graph.nodes.size();
            CopyNode *startWaitPtr = startWait.get();
            graph.nodes.emplace_back(std::move(startWait));
            graph.startWaitNodes[origin] = startWaitPtr;

            auto receiveWait = std::make_unique<CopyNode>();
            receiveWait->origin = origin;
            receiveWait->role = CopyNodeRole::RECEIVE_WAIT;
            receiveWait->index = graph.nodes.size();
            CopyNode *receiveWaitPtr = receiveWait.get();
            graph.nodes.emplace_back(std::move(receiveWait));
            graph.receiveWaitNodes[origin] = receiveWaitPtr;
            AddCopyEdge(startWaitPtr, receiveWaitPtr);
        } else {
            auto copy = std::make_unique<CopyNode>();
            copy->origin = origin;
            copy->index = graph.nodes.size();
            CopyNode *copyPtr = copy.get();
            graph.nodes.emplace_back(std::move(copy));
            graph.normalNodes[origin] = copyPtr;
        }
    }

    for (const TaskNode *parent : nodes) {
        for (const TaskNode *child : parent->GetChildren()) {
            CopyNode *parentCopy = FindCompletionCopy(graph, parent);
            if (parentCopy == nullptr) {
                return HCCL_E_INTERNAL;
            }
            if (IsWaitNode(child)) {
                const auto syncIter = syncParents.find(child);
                const bool isSyncParent = syncIter != syncParents.end() && syncIter->second.count(parent) != 0;
                if (!isSyncParent || IsSameRankStreamQueue(parent, child)) {
                    AddCopyEdge(parentCopy, graph.startWaitNodes.at(child));
                }
                if (isSyncParent) {
                    AddCopyEdge(parentCopy, graph.receiveWaitNodes.at(child));
                }
            } else {
                CopyNode *childCopy = FindCompletionCopy(graph, child);
                if (childCopy == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                AddCopyEdge(parentCopy, childCopy);
            }
        }
    }
    return HCCL_SUCCESS;
}

size_t CountCopyEdges(const CopiedGraph &graph)
{
    size_t count = 0;
    for (const auto &node : graph.nodes) {
        if (node != nullptr) {
            count += node->children.size();
        }
    }
    return count;
}

HcclResult BuildTopologicalOrder(const CopiedGraph &graph, std::vector<CopyNode *> &topoOrder)
{
    topoOrder.clear();
    std::vector<size_t> indegree(graph.nodes.size(), 0);
    for (const auto &node : graph.nodes) {
        if (node == nullptr) {
            return HCCL_E_PTR;
        }
        for (const CopyNode *child : node->children) {
            if (child == nullptr || child->index >= indegree.size()) {
                return HCCL_E_INTERNAL;
            }
            ++indegree[child->index];
        }
    }
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t index = 0; index < indegree.size(); ++index) {
        if (indegree[index] == 0) {
            ready.push(index);
        }
    }
    while (!ready.empty()) {
        const size_t index = ready.top();
        ready.pop();
        topoOrder.push_back(graph.nodes[index].get());
        for (const CopyNode *child : graph.nodes[index]->children) {
            if (--indegree[child->index] == 0) {
                ready.push(child->index);
            }
        }
    }
    if (topoOrder.size() != graph.nodes.size()) {
        HCCL_VM_ERROR("Sync conflict check failed because the copied sync DAG contains a cycle, "
            "copiedNodeCount={}, topoNodeCount={}", graph.nodes.size(), topoOrder.size());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

class ReachabilityChecker {
public:
    ReachabilityChecker(size_t nodeCount, const std::vector<size_t> &topoIndex)
        : nodeCount_(nodeCount), topoIndex_(topoIndex), visitStamp_(nodeCount, 0)
    {
    }

    bool IsReachable(const CopyNode *from, const CopyNode *to)
    {
        if (from == nullptr || to == nullptr || from->index >= nodeCount_ || to->index >= nodeCount_ ||
            topoIndex_[from->index] >= topoIndex_[to->index]) {
            return from == to;
        }
        if (queryStamp_ == std::numeric_limits<size_t>::max()) {
            std::fill(visitStamp_.begin(), visitStamp_.end(), 0);
            queryStamp_ = 1;
        } else {
            ++queryStamp_;
        }
        const size_t targetTopo = topoIndex_[to->index];
        std::vector<const CopyNode *> pending{from};
        visitStamp_[from->index] = queryStamp_;
        while (!pending.empty()) {
            const CopyNode *node = pending.back();
            pending.pop_back();
            for (const CopyNode *child : node->children) {
                if (child == to) {
                    return true;
                }
                if (child == nullptr || child->index >= nodeCount_ || topoIndex_[child->index] >= targetTopo ||
                    visitStamp_[child->index] == queryStamp_) {
                    continue;
                }
                visitStamp_[child->index] = queryStamp_;
                pending.push_back(child);
            }
        }
        return false;
    }

private:
    size_t nodeCount_{0};
    const std::vector<size_t> &topoIndex_;
    std::vector<size_t> visitStamp_;
    size_t queryStamp_{0};
};

bool HasMultiplePeer(const std::vector<SyncPair> &pairs, bool byPost)
{
    std::map<const CopyNode *, std::set<const CopyNode *>> peers;
    for (const SyncPair &pair : pairs) {
        if (pair.wait == nullptr || pair.receiveWaitCopy == nullptr) {
            continue;
        }
        const CopyNode *key = byPost ? pair.postCopy : pair.receiveWaitCopy;
        const CopyNode *peer = byPost ? pair.receiveWaitCopy : pair.postCopy;
        peers[key].insert(peer);
    }
    return std::any_of(peers.begin(), peers.end(), [](const auto &entry) {
        return entry.second.size() > 1U;
    });
}

HcclResult CheckBuckets(const SyncBuckets &buckets, ReachabilityChecker &reachable,
    SyncConflictCheckStats &stats)
{
    for (const auto &entry : buckets) {
        const SyncResourceKey &resource = entry.first;
        const SyncBucket &bucket = entry.second;
        const size_t postCount = bucket.originalPosts.size();
        const size_t waitCount = bucket.originalWaits.size();
        if (postCount < waitCount) {
            ++stats.checkedBucketCount;
            ++stats.conflictCount;
            HCCL_VM_ERROR("Sync resource has fewer producer tasks than consumer tasks, "
                "producerTaskType={}, consumerTaskType={}, resource={}, producerCount={}, consumerCount={}",
                SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind), DescribeResource(resource),
                postCount, waitCount);
            return HCCL_E_INTERNAL;
        }
        if (waitCount == 0U || bucket.pairs.size() <= 1U) {
            continue;
        }
        ++stats.checkedBucketCount;
        if (HasMultiplePeer(bucket.pairs, true) || HasMultiplePeer(bucket.pairs, false)) {
            ++stats.conflictCount;
            HCCL_VM_ERROR("Sync resource has multiple producer/consumer peers, "
                "producerTaskType={}, consumerTaskType={}, resource={}, producerCount={}, consumerCount={}, "
                "pairCount={}", SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind),
                DescribeResource(resource), postCount, waitCount, bucket.pairs.size());
            return HCCL_E_INTERNAL;
        }

        const SyncPair *lastPaired = nullptr;
        for (const SyncPair &pair : bucket.pairs) {
            if (pair.wait == nullptr) {
                if (lastPaired == nullptr) {
                    ++stats.conflictCount;
                    HCCL_VM_ERROR("Sync resource has an unmatched producer task before the first matched pair, "
                        "producerTaskType={}, consumerTaskType={}, resource={}, producer={}",
                        SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind),
                        DescribeResource(resource), pair.post->Describe());
                    return HCCL_E_INTERNAL;
                }
                if (!reachable.IsReachable(lastPaired->receiveWaitCopy, pair.postCopy)) {
                    ++stats.conflictCount;
                    HCCL_VM_ERROR("Sync resource has a many-to-one ordering conflict, "
                        "conflictType=many-to-one, producerTaskType={}, consumerTaskType={}, resource={}, "
                        "consumer={}, previousProducer={}, nextProducer={}",
                        SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind),
                        DescribeResource(resource), lastPaired->wait->Describe(), lastPaired->post->Describe(),
                        pair.post->Describe());
                    return HCCL_E_INTERNAL;
                }
                continue;
            }
            if (lastPaired != nullptr && lastPaired->receiveWaitCopy == nullptr) {
                ++stats.conflictCount;
                HCCL_VM_ERROR("Sync conflict checker has an invalid pair without a consumer completion node, "
                    "producerTaskType={}, consumerTaskType={}, resource={}, producer={}, consumer={}",
                    SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind),
                    DescribeResource(resource), lastPaired->post->Describe(), lastPaired->wait->Describe());
                return HCCL_E_INTERNAL;
            }
            if (lastPaired != nullptr && !reachable.IsReachable(lastPaired->receiveWaitCopy, pair.postCopy)) {
                ++stats.conflictCount;
                HCCL_VM_ERROR("Sync resource has a many-to-one ordering conflict, "
                    "conflictType=many-to-one, producerTaskType={}, consumerTaskType={}, resource={}, "
                    "consumer={}, previousProducer={}, nextProducer={}",
                    SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind),
                    DescribeResource(resource), lastPaired->wait->Describe(), lastPaired->post->Describe(),
                    pair.post->Describe());
                return HCCL_E_INTERNAL;
            }
            if (lastPaired != nullptr && !reachable.IsReachable(lastPaired->receiveWaitCopy, pair.startWaitCopy)) {
                ++stats.conflictCount;
                HCCL_VM_ERROR("Sync resource has a one-to-many ordering conflict, "
                    "conflictType=one-to-many, producerTaskType={}, consumerTaskType={}, resource={}, "
                    "producer={}, previousConsumer={}, nextConsumer={}",
                    SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind),
                    DescribeResource(resource), lastPaired->post->Describe(), lastPaired->wait->Describe(),
                    pair.wait->Describe());
                return HCCL_E_INTERNAL;
            }
            lastPaired = &pair;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult CheckSyncResourceConflict(const TaskNode *start, SyncConflictCheckStats *stats)
{
    SyncConflictCheckStats localStats;
    if (start == nullptr) {
        return HCCL_E_PTR;
    }

    std::vector<const TaskNode *> originalNodes;
    HcclResult ret = CollectOriginalNodes(start, originalNodes);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    localStats.originalNodeCount = originalNodes.size();

    SyncBuckets buckets;
    SyncParentMap syncParents;
    ret = BuildSyncPairs(originalNodes, buckets, syncParents);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    CopiedGraph copiedGraph;
    ret = BuildCopiedGraph(originalNodes, syncParents, copiedGraph);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    localStats.copiedNodeCount = copiedGraph.nodes.size();
    localStats.copiedEdgeCount = CountCopyEdges(copiedGraph);
    localStats.resourceBucketCount = buckets.size();

    for (auto &entry : buckets) {
        SyncBucket &bucket = entry.second;
        for (const TaskNode *post : bucket.originalPosts) {
            if (FindCompletionCopy(copiedGraph, post) == nullptr) {
                return HCCL_E_INTERNAL;
            }
        }
        for (SyncPair &pair : bucket.pairs) {
            pair.postCopy = FindCompletionCopy(copiedGraph, pair.post);
            pair.startWaitCopy = FindStartCopy(copiedGraph, pair.wait);
            pair.receiveWaitCopy = FindCompletionCopy(copiedGraph, pair.wait);
            if (pair.postCopy == nullptr || pair.startWaitCopy == nullptr || pair.receiveWaitCopy == nullptr) {
                return HCCL_E_INTERNAL;
            }
        }
        std::set<const TaskNode *> pairedPosts;
        for (const SyncPair &pair : bucket.pairs) {
            pairedPosts.insert(pair.post);
        }
        for (const TaskNode *post : bucket.originalPosts) {
            if (pairedPosts.count(post) == 0U) {
                SyncPair residual;
                residual.resource = entry.first;
                residual.post = post;
                residual.postCopy = FindCompletionCopy(copiedGraph, post);
                if (residual.postCopy == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                bucket.pairs.push_back(residual);
            }
        }
        localStats.pairCount += bucket.pairs.size();
    }

    std::vector<CopyNode *> topoOrder;
    ret = BuildTopologicalOrder(copiedGraph, topoOrder);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    std::vector<size_t> topoIndex(copiedGraph.nodes.size(), 0);
    for (size_t index = 0; index < topoOrder.size(); ++index) {
        topoIndex[topoOrder[index]->index] = index;
    }
    for (auto &entry : buckets) {
        auto &pairs = entry.second.pairs;
        std::sort(pairs.begin(), pairs.end(), [&topoIndex](const SyncPair &lhs, const SyncPair &rhs) {
            const size_t lhsPost = topoIndex[lhs.postCopy->index];
            const size_t rhsPost = topoIndex[rhs.postCopy->index];
            if (lhsPost != rhsPost) {
                return lhsPost < rhsPost;
            }
            const size_t lhsWait = lhs.receiveWaitCopy == nullptr ? std::numeric_limits<size_t>::max() :
                topoIndex[lhs.receiveWaitCopy->index];
            const size_t rhsWait = rhs.receiveWaitCopy == nullptr ? std::numeric_limits<size_t>::max() :
                topoIndex[rhs.receiveWaitCopy->index];
            return lhsWait < rhsWait;
        });
        for (size_t index = 0; index < pairs.size(); ++index) {
            pairs[index].order = index + 1U;
        }
    }

    ReachabilityChecker reachable(copiedGraph.nodes.size(), topoIndex);
    ret = CheckBuckets(buckets, reachable, localStats);
    if (stats != nullptr) {
        *stats = localStats;
    }
    return ret;
}
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
