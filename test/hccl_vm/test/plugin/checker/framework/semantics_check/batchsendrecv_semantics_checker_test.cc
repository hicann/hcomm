/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <set>

#include "batchsendrecv_semantics_checker.h"
#include "check_utils.h"

namespace HcclSim {
class BatchSendRecvSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {
    }
    
    void TearDown() override {
    }

    // Helper function to create valid BatchSendRecv semantics
    void CreateValidBatchSendRecvSemantics(
        std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
        u32 rankSize, u64 dataSize) {
        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            RankMemorySemantics rankMemSemantics;
            std::set<BufferSemantic> outputSemantics;
            
            u64 totalSize = 0;
            for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
                BufferSemantic bufSem(totalSize, dataSize);
                bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, rankId * dataSize));
                outputSemantics.insert(bufSem);
                totalSize += dataSize;
            }
            
            rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
            allRankMemSemantics[rankId] = rankMemSemantics;
        }
    }
};

// Test normal case: valid BatchSendRecv semantics with 2 ranks
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_TwoRanks) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    
    CreateValidBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    
    HcclResult result = TaskCheckBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid BatchSendRecv semantics with 4 ranks
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_FourRanks) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    
    CreateValidBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    
    HcclResult result = TaskCheckBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_SingleRank) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;
    
    CreateValidBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    
    HcclResult result = TaskCheckBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size - manually construct zero-size BufferSemantic
// to satisfy checker's curRankId advancement logic (empty set would fail curRankId != rankSize)
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_ZeroDataSize) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 1;
    u64 dataSize = 0;

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        RankMemorySemantics rankMemSemantics;
        std::set<BufferSemantic> outputSemantics;

        u64 totalSize = 0;
        for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
            BufferSemantic bufSem(totalSize, dataSize);
            bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, dataSize * rankId));
            outputSemantics.insert(bufSem);
            totalSize += dataSize;
        }

        rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
        allRankMemSemantics[rankId] = rankMemSemantics;
    }

    HcclResult result = TaskCheckBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(BatchSendRecvSemanticsCheckerTest, Abnormal_MissingRank) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    
    // Only add rank 0
    RankMemorySemantics rankMemSemantics;
    std::set<BufferSemantic> outputSemantics;
    
    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    outputSemantics.insert(bufSem0);
    
    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, dataSize));
    outputSemantics.insert(bufSem1);
    
    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    
    HcclResult result = TaskCheckBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST_F(BatchSendRecvSemanticsCheckerTest, Abnormal_ExpectedRankSetIsNotInferredFromActualMap) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    constexpr u64 dataSize = 1024;

    RankMemorySemantics rankMemSemantics;
    BufferSemantic output(0, dataSize);
    output.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    rankMemSemantics[BufferType::OUTPUT].insert(output);
    allRankMemSemantics[0] = rankMemSemantics;

    EXPECT_EQ(TaskCheckBatchSendRecvSemantics(allRankMemSemantics, 2, dataSize),
        HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank
TEST_F(BatchSendRecvSemanticsCheckerTest, Abnormal_WrongSourceRank) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    
    RankMemorySemantics rankMemSemantics;
    std::set<BufferSemantic> outputSemantics;
    
    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.insert(bufSem0);
    
    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, dataSize)); // Wrong source rank
    outputSemantics.insert(bufSem1);
    
    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;
    
    HcclResult result = TaskCheckBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong buffer type
TEST_F(BatchSendRecvSemanticsCheckerTest, Abnormal_WrongBufferType) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    
    RankMemorySemantics rankMemSemantics;
    std::set<BufferSemantic> outputSemantics;
    
    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::OUTPUT, 0)); // Wrong buffer type
    outputSemantics.insert(bufSem0);
    
    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, dataSize));
    outputSemantics.insert(bufSem1);
    
    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;
    
    HcclResult result = TaskCheckBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large number of ranks
TEST_F(BatchSendRecvSemanticsCheckerTest, ValidSemantics_LargeRankSize) {
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;
    
    CreateValidBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    
    HcclResult result = TaskCheckBatchSendRecvSemantics(allRankMemSemantics, rankSize, dataSize);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
}
