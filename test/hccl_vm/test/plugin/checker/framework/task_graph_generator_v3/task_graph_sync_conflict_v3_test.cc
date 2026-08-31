/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "task_graph_sync_conflict_v3.h"

namespace HcclSim {
namespace {
using namespace TaskGraphGeneratorV3;

class SyncConflictTest : public testing::Test {
protected:
    template <typename T>
    T *AddNode(std::unique_ptr<T> node, const TaskPosition &position = TaskPosition{})
    {
        T *raw = node.get();
        raw->SetNodeId(static_cast<NodeId>(nodes_.size()));
        raw->SetPosition(position);
        nodes_.emplace_back(std::move(node));
        return raw;
    }

    TaskStart *AddMainStart()
    {
        auto node = std::make_unique<TaskStart>(BoundaryType::MAIN_GRAPH);
        node->SetNodeId(MAIN_START_NODE_ID);
        TaskStart *raw = node.get();
        nodes_.emplace_back(std::move(node));
        return raw;
    }

    static AicpuNotify MakeNotify(uint32_t notifyId)
    {
        AicpuNotify notify;
        notify.recordRankId = 0;
        notify.waitRankId = 0;
        notify.notifyId = notifyId;
        return notify;
    }

    TaskRecordAICPU *AddAicpuPost(uint32_t notifyId, const TaskPosition &position = TaskPosition{})
    {
        return AddNode(std::make_unique<TaskRecordAICPU>(MakeNotify(notifyId), ProtocolType::SDMA), position);
    }

    TaskWaitAICPU *AddAicpuWait(uint32_t notifyId, const TaskPosition &position = TaskPosition{})
    {
        return AddNode(std::make_unique<TaskWaitAICPU>(MakeNotify(notifyId), ProtocolType::SDMA), position);
    }

    TaskRecordCCU *AddCcuPost(uint16_t mask, RankId waitRankId = 0, uint32_t dieId = 0,
        uint16_t ckeId = 1, const TaskPosition &position = TaskPosition{})
    {
        CcuNotify notify;
        notify.waitRankId = waitRankId;
        notify.dieId = dieId;
        notify.ckeId = ckeId;
        notify.ckeMask = mask;
        return AddNode(std::make_unique<TaskRecordCCU>(notify, ProtocolType::CCU), position);
    }

    TaskWaitCCU *AddCcuWait(uint16_t mask, RankId waitRankId = 0, uint32_t dieId = 0,
        uint16_t ckeId = 1, const TaskPosition &position = TaskPosition{})
    {
        CcuNotify notify;
        notify.waitRankId = waitRankId;
        notify.dieId = dieId;
        notify.ckeId = ckeId;
        notify.ckeMask = mask;
        return AddNode(std::make_unique<TaskWaitCCU>(notify, ProtocolType::CCU), position);
    }

    static void AddEdge(TaskNode *parent, TaskNode *child)
    {
        ASSERT_NE(parent, nullptr);
        ASSERT_NE(child, nullptr);
        ASSERT_TRUE(parent->AddChild(child));
        ASSERT_TRUE(child->AddParent(parent));
    }

    std::vector<std::unique_ptr<TaskNode>> nodes_;
};

TEST_F(SyncConflictTest, AicpuPairIsAcceptedAndDoesNotModifyGraph)
{
    TaskStart *start = AddMainStart();
    TaskRecordAICPU *post = AddAicpuPost(1);
    TaskWaitAICPU *wait = AddAicpuWait(1);
    AddEdge(start, post);
    AddEdge(post, wait);

    const std::vector<TaskNode *> children = post->GetChildren();
    SyncConflictCheckStats stats;
    EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
    EXPECT_EQ(stats.resourceBucketCount, 1U);
    EXPECT_EQ(stats.pairCount, 1U);
    EXPECT_EQ(stats.conflictCount, 0U);
    EXPECT_EQ(post->GetChildren(), children);
}

TEST_F(SyncConflictTest, AicpuMultiplePeersAreRejected)
{
    TaskStart *start = AddMainStart();
    TaskRecordAICPU *post = AddAicpuPost(1);
    TaskWaitAICPU *wait1 = AddAicpuWait(1);
    TaskWaitAICPU *wait2 = AddAicpuWait(1);
    AddEdge(start, post);
    AddEdge(post, wait1);
    AddEdge(post, wait2);

    SyncConflictCheckStats stats;
    EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
    EXPECT_EQ(stats.checkedBucketCount, 1U);
    EXPECT_EQ(stats.conflictCount, 1U);
}

TEST_F(SyncConflictTest, OrderedAicpuReuseIsAccepted)
{
    TaskStart *start = AddMainStart();
    TaskRecordAICPU *post1 = AddAicpuPost(1);
    TaskWaitAICPU *wait1 = AddAicpuWait(1);
    TaskRecordAICPU *post2 = AddAicpuPost(1);
    TaskWaitAICPU *wait2 = AddAicpuWait(1);
    AddEdge(start, post1);
    AddEdge(post1, wait1);
    AddEdge(wait1, post2);
    AddEdge(post2, wait2);

    SyncConflictCheckStats stats;
    EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
    EXPECT_EQ(stats.pairCount, 2U);
    EXPECT_EQ(stats.checkedBucketCount, 1U);
    EXPECT_EQ(stats.conflictCount, 0U);
}

TEST_F(SyncConflictTest, CcuMaskBitsAreIndependentResources)
{
    TaskStart *start = AddMainStart();
    TaskRecordCCU *post = AddCcuPost(0x3);
    TaskWaitCCU *wait = AddCcuWait(0x3);
    AddEdge(start, post);
    AddEdge(post, wait);

    SyncConflictCheckStats stats;
    EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
    EXPECT_EQ(stats.resourceBucketCount, 2U);
    EXPECT_EQ(stats.pairCount, 2U);
    EXPECT_EQ(stats.conflictCount, 0U);
}

TEST_F(SyncConflictTest, CcuMultiplePeersAreRejected)
{
    TaskStart *start = AddMainStart();
    TaskRecordCCU *post1 = AddCcuPost(0x1);
    TaskRecordCCU *post2 = AddCcuPost(0x1);
    TaskWaitCCU *wait = AddCcuWait(0x1);
    AddEdge(start, post1);
    AddEdge(start, post2);
    AddEdge(post1, wait);
    AddEdge(post2, wait);

    SyncConflictCheckStats stats;
    EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
    EXPECT_EQ(stats.resourceBucketCount, 1U);
    EXPECT_EQ(stats.pairCount, 2U);
    EXPECT_EQ(stats.checkedBucketCount, 1U);
    EXPECT_EQ(stats.conflictCount, 1U);
}

TEST_F(SyncConflictTest, CcuResourceFieldsAreIndependent)
{
    TaskStart *start = AddMainStart();
    const TaskPosition lane{0, 0, 0};
    const std::vector<std::tuple<RankId, uint32_t, uint16_t>> resources{
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {0, 0, 2}};
    for (const auto &resource : resources) {
        TaskRecordCCU *post = AddCcuPost(1, std::get<0>(resource), std::get<1>(resource),
            std::get<2>(resource), lane);
        TaskWaitCCU *wait = AddCcuWait(1, std::get<0>(resource), std::get<1>(resource),
            std::get<2>(resource), lane);
        AddEdge(start, post);
        AddEdge(post, wait);
    }

    SyncConflictCheckStats stats;
    EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
    EXPECT_EQ(stats.resourceBucketCount, resources.size());
    EXPECT_EQ(stats.pairCount, resources.size());
}

TEST_F(SyncConflictTest, InvalidResourcesAreIgnored)
{
    TaskStart *start = AddMainStart();
    TaskRecordAICPU *aicpuPost = AddAicpuPost(INVALID_NOTIFY_ID);
    TaskWaitAICPU *aicpuWait = AddAicpuWait(INVALID_NOTIFY_ID);
    TaskRecordCCU *ccuPost = AddCcuPost(0);
    TaskWaitCCU *ccuWait = AddCcuWait(0);
    AddEdge(start, aicpuPost);
    AddEdge(aicpuPost, aicpuWait);
    AddEdge(start, ccuPost);
    AddEdge(ccuPost, ccuWait);

    SyncConflictCheckStats stats;
    EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
    EXPECT_EQ(stats.resourceBucketCount, 0U);
    EXPECT_EQ(stats.pairCount, 0U);
}

TEST_F(SyncConflictTest, NullStartIsRejected)
{
    SyncConflictCheckStats stats;
    EXPECT_EQ(CheckSyncResourceConflict(nullptr, &stats), HCCL_E_PTR);
    EXPECT_EQ(stats.originalNodeCount, 0U);
}
} // namespace
} // namespace HcclSim

