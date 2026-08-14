/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "hccl_nslbdp.h"
#include "hccl_nslbdp_pub.h"
#include "hccl_nslb_md5.h"

using namespace hccl;
using namespace testing;

namespace {
constexpr u32 IP_OCTET_0_SHIFT = 24;
constexpr u32 IP_OCTET_1_SHIFT = 16;
constexpr u32 IP_OCTET_2_SHIFT = 8;
constexpr u32 IP_BYTE_MASK = 0xFF;
constexpr u32 COMM_DESC_OVERFLOW_LEN = COMM_DESC_MAX_LENGTH - 1; // strncpy_s count 不得超过 destMax，否则 abort
constexpr u64 DUMMY_TASK_ID = 0xABCDULL;

inline u32 BuildIpv4(u32 a, u32 b, u32 c, u32 d)
{
    return ((a & IP_BYTE_MASK) << IP_OCTET_0_SHIFT) | ((b & IP_BYTE_MASK) << IP_OCTET_1_SHIFT)
           | ((c & IP_BYTE_MASK) << IP_OCTET_2_SHIFT) | (d & IP_BYTE_MASK);
}

std::string MakeLongString(size_t len, char fill) { return std::string(len, fill); }

NslbDpCommConfigVal BuildCommCfg(const char* desc, const u8* md5Bytes = nullptr)
{
    NslbDpCommConfigVal cfg = {};
    cfg.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(cfg.commDesc, COMM_DESC_MAX_LENGTH, desc, COMM_DESC_MAX_LENGTH - 1);
    cfg.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    if (md5Bytes != nullptr) {
        (void)memcpy_s(cfg.commMd5Sum, sizeof(cfg.commMd5Sum), md5Bytes, sizeof(cfg.commMd5Sum));
    }
    return cfg;
}
} // namespace

// ============================================================
// 1. 纯函数：GetNslbOpType
// ============================================================
TEST(HcclNslbDpPureFuncTest, GetNslbOpType_AllKnownCases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_BROADCAST), NSLBDP_CMD_BROADCAST);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLREDUCE), NSLBDP_CMD_ALLREDUCE);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_REDUCE), NSLBDP_CMD_REDUCE);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_SEND), NSLBDP_CMD_SEND);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_RECEIVE), NSLBDP_CMD_RECEIVE);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLGATHER), NSLBDP_CMD_ALLGATHER);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_REDUCE_SCATTER), NSLBDP_CMD_REDUCE_SCATTER);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLTOALLV), NSLBDP_CMD_ALLTOALLV);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLTOALLVC), NSLBDP_CMD_ALLTOALLVC);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLTOALL), NSLBDP_CMD_ALLTOALL);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_GATHER), NSLBDP_CMD_GATHER);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_SCATTER), NSLBDP_CMD_SCATTER);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV), NSLBDP_CMD_BATCH_SEND_RECV);
    EXPECT_EQ(inst.GetNslbOpType(static_cast<HcclCMDType>(0xDEAD)), 0); // default 分支
}

// ============================================================
// 2. 纯函数：GetNslbLevel1AlgType / Level2
// ============================================================
TEST(HcclNslbDpPureFuncTest, GetNslbLevel1AlgType_Cases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_RING), NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_PIPELINE), NSLB_ALGO_TYPE_PIPELINE);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_HD), NSLB_ALGO_TYPE_HDR);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_NHR), NSLB_ALGO_TYPE_NHR);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_NHR_V1), NSLB_ALGO_TYPE_NHR_V1);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_NB), NSLB_ALGO_TYPE_NB);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_AHC), NSLB_ALGO_TYPE_AHC);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE), NSLB_ALGO_TYPE_AHC);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(static_cast<AlgTypeLevel1>(0xFF)), NSLB_ALGO_TYPE_NA); // default
}

TEST(HcclNslbDpPureFuncTest, GetNslbLevel2AlgType_Cases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_RING), NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_HD), NSLB_ALGO_TYPE_HDR);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_NHR), NSLB_ALGO_TYPE_NHR);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_NB), NSLB_ALGO_TYPE_NB);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_PIPELINE), NSLB_ALGO_TYPE_PIPELINE);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(static_cast<AlgTypeLevel2>(0xFF)), NSLB_ALGO_TYPE_NA); // default
}

// ============================================================
// 3. 纯函数：GetNslbDpFirstFourBit
// ============================================================
TEST(HcclNslbDpPureFuncTest, GetNslbDpFirstFourBit_ByOptype)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    // ALLREDUCE: (1<<2)+1 = 5
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_ALLREDUCE, NSLB_ALGO_TYPE_RING), 5ULL);
    // ALLGATHER: (1<<1) = 2
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_ALLGATHER, NSLB_ALGO_TYPE_RING), 2ULL);
    // 其它 op 均返回 0
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_REDUCE_SCATTER, NSLB_ALGO_TYPE_RING), 0ULL);
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_ALLTOALL, NSLB_ALGO_TYPE_RING), 0ULL);
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_BROADCAST, NSLB_ALGO_TYPE_AHC), 0ULL);
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(0, 0), 0ULL);
}

// ============================================================
// 4. 纯函数：CheckSupportOptype
// ============================================================
TEST(HcclNslbDpPureFuncTest, CheckSupportOptype_TrueCases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLREDUCE));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLTOALL));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLTOALLV));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLTOALLVC));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_BROADCAST));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_SCATTER));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_REDUCE));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_SEND));
}

TEST(HcclNslbDpPureFuncTest, CheckSupportOptype_FalseCases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    // GATHER / RECEIVE / BATCH_PUT / BATCH_GET 不在支持列表
    EXPECT_FALSE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_GATHER));
    EXPECT_FALSE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_RECEIVE));
    EXPECT_FALSE(inst.CheckSupportOptype(static_cast<HcclCMDType>(0xDEAD)));
}

// ============================================================
// 5. 纯函数：CheckAlgoConsistency
// ============================================================
TEST(HcclNslbDpPureFuncTest, CheckAlgoConsistency_AllReduce)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    std::string hit = "RunAllReduceRingV1";
    std::string miss = "RunAllGatherRingV1";
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLREDUCE, hit));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLREDUCE, miss));
}

TEST(HcclNslbDpPureFuncTest, CheckAlgoConsistency_OtherOps)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    std::string ag = "DoAllGatherPipeline";
    std::string rs = "DoReduceScatterRing";
    std::string a2aFull = "RunAlltoAllVFullMesh";
    std::string a2aOther = "AlltoAllVOtherName";
    std::string bc = "BroadCastTree";
    std::string sc = "ScatterBinary";
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLGATHER, ag));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLGATHER, rs));
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, rs));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ag));
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLTOALL, a2aFull));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLTOALL, a2aOther));
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_BROADCAST, bc));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_BROADCAST, ag));
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_SCATTER, sc));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_SCATTER, ag));
}

// ============================================================
// 6. 纯函数：ipToUint32
// ============================================================
TEST(HcclNslbDpPureFuncTest, IpToUint32_Normal)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.ipToUint32("127.0.0.1"), BuildIpv4(127, 0, 0, 1));
    EXPECT_EQ(inst.ipToUint32("0.0.0.0"), 0U);
    EXPECT_EQ(inst.ipToUint32("255.255.255.255"), 0xFFFFFFFFU);
    EXPECT_EQ(inst.ipToUint32("192.168.1.1"), BuildIpv4(192, 168, 1, 1));
    EXPECT_EQ(inst.ipToUint32("10.11.12.13"), BuildIpv4(10, 11, 12, 13));
}

// ============================================================
// 7. 纯函数：SplitString
// ============================================================
TEST(HcclNslbDpPureFuncTest, SplitString_Normal)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    std::vector<std::string> out;
    inst.SplitString("a_b_c", out, "_");
    ASSERT_EQ(out.size(), 3U);
    EXPECT_EQ(out[0], "a");
    EXPECT_EQ(out[1], "b");
    EXPECT_EQ(out[2], "c");
}

TEST(HcclNslbDpPureFuncTest, SplitString_Edge)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    std::vector<std::string> out1;
    inst.SplitString("noseparator", out1, "_");
    ASSERT_EQ(out1.size(), 1U);
    EXPECT_EQ(out1[0], "noseparator");

    std::vector<std::string> out2;
    inst.SplitString("", out2, "_");
    EXPECT_EQ(out2.size(), 0U); // 空串 → 不 push_back（pos1==length）

    std::vector<std::string> out3;
    inst.SplitString("a__b", out3, "_"); // 连续分隔符产生空段
    ASSERT_EQ(out3.size(), 3U);
    EXPECT_EQ(out3[0], "a");
    EXPECT_EQ(out3[1], "");
    EXPECT_EQ(out3[2], "b");
}

// ============================================================
// 8. 弱状态：fullcommDescInitTime
// ============================================================
TEST(HcclNslbDpWeakStateTest, FullcommDescInitTime_WithAndWithoutTimestamp)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    NslbDpOperatorInfo op = {};

    // 重构后行为：不再截掉时间戳，直接原样拷贝 identifier
    std::string withTs = "aaa_bbb_ccc_1710000000123";
    inst.fullcommDescInitTime(withTs, op);
    EXPECT_STREQ(op.commDesc, withTs.c_str());
    EXPECT_EQ(op.commDesc[COMM_DESC_MAX_LENGTH - 1], '\0');
    EXPECT_NE(op.commInitTime, 0ULL);

    std::string withoutTs = "aaa_bbb_ccc_ddd_eee";
    NslbDpOperatorInfo op2 = {};
    inst.fullcommDescInitTime(withoutTs, op2);
    EXPECT_STREQ(op2.commDesc, withoutTs.c_str());
    EXPECT_NE(op2.commInitTime, 0ULL);
}

// ============================================================
// 9. 弱状态 + 注入表项：CheckCommDescExit / CheckSameOperatorVal
// ============================================================
TEST(HcclNslbDpWeakStateTest, CheckCommDescExit_InjectedConfig)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    // 备份并清空，避免其它 UT 残留
    auto backup = std::move(inst.hcclNslbDpCommConfig_);
    inst.hcclNslbDpCommConfig_.clear();

    NslbDpOperatorInfo op = {};
    (void)strncpy_s(op.commDesc, COMM_DESC_MAX_LENGTH, "exist_group", COMM_DESC_MAX_LENGTH - 1);
    EXPECT_FALSE(inst.CheckCommDescExit(op));

    inst.hcclNslbDpCommConfig_.push_back(BuildCommCfg("exist_group"));
    EXPECT_TRUE(inst.CheckCommDescExit(op));

    NslbDpOperatorInfo op2 = {};
    (void)strncpy_s(op2.commDesc, COMM_DESC_MAX_LENGTH, "ghost", COMM_DESC_MAX_LENGTH - 1);
    EXPECT_FALSE(inst.CheckCommDescExit(op2));

    inst.hcclNslbDpCommConfig_ = std::move(backup);
}

TEST(HcclNslbDpWeakStateTest, CheckSameOperatorVal_InjectedVal)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpOperatorVal_);
    inst.hcclNslbDpOperatorVal_.clear();

    NslbDpOperatorInfo op = {};
    op.taskId = DUMMY_TASK_ID;
    op.rootRank = 123;
    op.oper = NSLBDP_CMD_ALLREDUCE;
    op.algorithm = NSLB_ALGO_TYPE_RING;
    (void)strncpy_s(op.commDesc, COMM_DESC_MAX_LENGTH, "g", COMM_DESC_MAX_LENGTH - 1);

    // operSize >= num → false
    EXPECT_FALSE(inst.CheckSameOperatorVal(0, op, 123));

    inst.hcclNslbDpOperatorVal_.push_back(op);
    EXPECT_TRUE(inst.CheckSameOperatorVal(0, op, 123));

    // rootRank 不同 → false
    EXPECT_FALSE(inst.CheckSameOperatorVal(0, op, 999));
    // commDesc 不同 → false
    op.commDesc[0] = 'x';
    EXPECT_FALSE(inst.CheckSameOperatorVal(0, op, 123));

    inst.hcclNslbDpOperatorVal_ = std::move(backup);
}

// ============================================================
// 10. 拆分助手 1：InitAlgInfoCommDesc
// ============================================================
TEST(HcclNslbDpHelpersTest, InitAlgInfoCommDesc_ExistAndTruncate)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpCommConfig_);
    inst.hcclNslbDpCommConfig_.clear();

    NslbDpAlgorithmInfo a = {};
    // 恰好 COMM_DESC_MAX_LENGTH-1 个字符，验证 strncpy_s 全量拷贝与 NUL 终止
    // 注意：strncpy_s 的 count 参数不得超过 destMax，否则安全函数库会 abort
    std::string longId = MakeLongString(COMM_DESC_OVERFLOW_LEN, 'x');
    EXPECT_FALSE(inst.InitAlgInfoCommDesc(a, longId)); // 表空 → 不存在
    EXPECT_EQ(a.commDesc[COMM_DESC_MAX_LENGTH - 1], '\0');
    for (u32 i = 0; i < COMM_DESC_OVERFLOW_LEN; i++) {
        EXPECT_EQ(a.commDesc[i], 'x');
    }

    inst.hcclNslbDpCommConfig_.push_back(BuildCommCfg("exactName"));
    NslbDpAlgorithmInfo b = {};
    EXPECT_TRUE(inst.InitAlgInfoCommDesc(b, std::string("exactName")));

    NslbDpAlgorithmInfo c = {};
    EXPECT_FALSE(inst.InitAlgInfoCommDesc(c, std::string("otherName")));

    inst.hcclNslbDpCommConfig_ = std::move(backup);
}

// ============================================================
// 11. 拆分助手 2：FillAlgInfoCommMd5
// ============================================================
TEST(HcclNslbDpHelpersTest, FillAlgInfoCommMd5_FoundAndNotFound)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpCommConfig_);
    inst.hcclNslbDpCommConfig_.clear();

    u8 goldenMd5[NSLB_MD5_DIGEST];
    for (u32 i = 0; i < NSLB_MD5_DIGEST; i++) {
        goldenMd5[i] = static_cast<u8>(i + 1U);
    }
    inst.hcclNslbDpCommConfig_.push_back(BuildCommCfg("withMd5", goldenMd5));

    NslbDpAlgorithmInfo a = {};
    (void)strncpy_s(a.commDesc, COMM_DESC_MAX_LENGTH, "withMd5", COMM_DESC_MAX_LENGTH - 1);
    EXPECT_TRUE(inst.FillAlgInfoCommMd5(a));
    EXPECT_EQ(memcmp(a.commMd5Sum, goldenMd5, NSLB_MD5_DIGEST), 0);

    NslbDpAlgorithmInfo b = {};
    (void)strncpy_s(b.commDesc, COMM_DESC_MAX_LENGTH, "noSuch", COMM_DESC_MAX_LENGTH - 1);
    // 找不到不进 memcpy，返回 true（不报错，只是不填充）
    u8 zero[NSLB_MD5_DIGEST] = {0};
    EXPECT_TRUE(inst.FillAlgInfoCommMd5(b));
    EXPECT_EQ(memcmp(b.commMd5Sum, zero, NSLB_MD5_DIGEST), 0);

    inst.hcclNslbDpCommConfig_ = std::move(backup);
}

// ============================================================
// 12. 拆分助手 3：FillAlgInfoBaseFields（纯赋值）
// ============================================================
TEST(HcclNslbDpHelpersTest, FillAlgInfoBaseFields_PureAssignment)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    NslbDpAlgorithmInfo a = {};
    inst.FillAlgInfoBaseFields(a, HcclCMDType::HCCL_CMD_ALLGATHER, 7U, 3U, NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(a.srcLocalRankId, 7U);
    EXPECT_EQ(a.rootRank, 3U);
    EXPECT_EQ(a.oper, NSLBDP_CMD_ALLGATHER);
    EXPECT_EQ(a.algorithm, NSLB_ALGO_TYPE_RING);

    NslbDpAlgorithmInfo b = {};
    inst.FillAlgInfoBaseFields(b, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, 0xFFFFFFFFU, 0xFFFEU, NSLB_ALGO_TYPE_AHC);
    EXPECT_EQ(b.srcLocalRankId, static_cast<u16>(0xFFFFFFFFU));
    EXPECT_EQ(b.rootRank, 0xFFFEU);
    EXPECT_EQ(b.oper, NSLBDP_CMD_BATCH_SEND_RECV);
    EXPECT_EQ(b.algorithm, NSLB_ALGO_TYPE_AHC);
}

// ============================================================
// 13. 拆分助手 4：IsAlgAdjacencyDuplicated
// ============================================================
TEST(HcclNslbDpHelpersTest, IsAlgAdjacencyDuplicated_BothBranches)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpAlgorithmInfo_);
    inst.hcclNslbDpAlgorithmInfo_.clear();

    NslbDpAlgorithmInfo base = {};
    base.taskId = DUMMY_TASK_ID;
    base.srcLocalRankId = 2;
    base.rootRank = 5;
    base.oper = NSLBDP_CMD_ALLREDUCE;
    base.algorithm = NSLB_ALGO_TYPE_RING;
    (void)strncpy_s(base.commDesc, COMM_DESC_MAX_LENGTH, "dup_test", COMM_DESC_MAX_LENGTH - 1);

    EXPECT_FALSE(inst.IsAlgAdjacencyDuplicated(base)); // 空表 → 不重复

    inst.hcclNslbDpAlgorithmInfo_.push_back(base);
    EXPECT_TRUE(inst.IsAlgAdjacencyDuplicated(base)); // 全字段相等 → 重复

    NslbDpAlgorithmInfo diff = base;
    diff.srcLocalRankId = 9;
    EXPECT_FALSE(inst.IsAlgAdjacencyDuplicated(diff)); // srcLocalRankId 不同 → 不重复

    NslbDpAlgorithmInfo diffDesc = base;
    (void)strncpy_s(diffDesc.commDesc, COMM_DESC_MAX_LENGTH, "other", COMM_DESC_MAX_LENGTH - 1);
    EXPECT_FALSE(inst.IsAlgAdjacencyDuplicated(diffDesc));

    inst.hcclNslbDpAlgorithmInfo_ = std::move(backup);
}

// ============================================================
// 14. 拆分助手 5：FillAlgInfoAdjInfo（空/非空分支）
// ============================================================
TEST(HcclNslbDpHelpersTest, FillAlgInfoAdjInfo_EmptyAndNonEmpty)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();

    AdjInfo emptyAdj = {};
    emptyAdj.dstRankNum = 99;
    emptyAdj.nsAdjInfo.clear();
    NslbDpAlgorithmInfo a = {};
    EXPECT_FALSE(inst.FillAlgInfoAdjInfo(a, emptyAdj, 1U));
    EXPECT_EQ(a.dstRankNum, 0U); // 被清零

    AdjInfo adj;
    adj.dstRankNum = 3;
    NslbDpAdjInfo r1 = {10, 1, 0};
    NslbDpAdjInfo r2 = {20, 2, 0};
    NslbDpAdjInfo r3 = {30, 3, 0};
    adj.nsAdjInfo.push_back(r1);
    adj.nsAdjInfo.push_back(r2);
    adj.nsAdjInfo.push_back(r3);
    NslbDpAlgorithmInfo b = {};
    EXPECT_TRUE(inst.FillAlgInfoAdjInfo(b, adj, 7U));
    EXPECT_EQ(b.dstRankNum, 3U);
    ASSERT_EQ(b.AdjInfo.size(), 3U);
    EXPECT_EQ(b.AdjInfo[0].dstLocalRankId, 10U);
    EXPECT_EQ(b.AdjInfo[0].phaseId, 1U);
    EXPECT_EQ(b.AdjInfo[0].rev, 0U);
    EXPECT_EQ(b.AdjInfo[1].dstLocalRankId, 20U);
    EXPECT_EQ(b.AdjInfo[1].phaseId, 2U);
    EXPECT_EQ(b.AdjInfo[2].dstLocalRankId, 30U);
    EXPECT_EQ(b.AdjInfo[2].phaseId, 3U);
}

// ============================================================
// 15. ST 流程：SetGlobalCommTaskId 使能门控 + 注入 commConfig
// ============================================================
namespace {
constexpr char kTestCommDesc[] = "st_flow_group";

class NslbDpStFlowTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        inst_ = &hcclNslbDp::GetInstance();
        inst_->SetGlobalCommTaskId(DUMMY_TASK_ID);

        // 注入 commConfig 表项（ST 初始化通信域时会做这件事）
        commCfgBackup_ = std::move(inst_->hcclNslbDpCommConfig_);
        inst_->hcclNslbDpCommConfig_.clear();
        inst_->hcclNslbDpCommConfig_.push_back(BuildCommCfg(kTestCommDesc));

        // 隔离其他内部状态
        algorithmInfoBackup_ = std::move(inst_->hcclNslbDpAlgorithmInfo_);
        operatorValBackup_ = std::move(inst_->hcclNslbDpOperatorVal_);
        netcoFlagBackup_ = inst_->nslbdpIsInitNetCo_.load();
        l4sPortIdBackup_ = inst_->hcclNslbDpL4SPortId_;
    }

    void TearDown() override
    {
        inst_->hcclNslbDpCommConfig_ = std::move(commCfgBackup_);
        inst_->hcclNslbDpAlgorithmInfo_ = std::move(algorithmInfoBackup_);
        inst_->hcclNslbDpOperatorVal_ = std::move(operatorValBackup_);
        inst_->nslbdpIsInitNetCo_ = netcoFlagBackup_;
        inst_->hcclNslbDpL4SPortId_ = l4sPortIdBackup_;
        inst_->SetGlobalCommTaskId(0);
    }

    hcclNslbDp* inst_;
    std::vector<NslbDpCommConfigVal> commCfgBackup_;
    std::vector<NslbDpAlgorithmInfo> algorithmInfoBackup_;
    std::vector<NslbDpOperatorInfo> operatorValBackup_;
    bool netcoFlagBackup_;
    u32 l4sPortIdBackup_;
};
} // namespace

// ============================================================
// 16. GetAlgAdjacencyTable：ST 全路径（非零 taskId + commConfig + 有效 AdjInfo）
// ============================================================
TEST_F(NslbDpStFlowTest, GetAlgAdjacencyTable_FullPath)
{
    AdjInfo adjInfo = {};
    adjInfo.dstRankNum = 3;
    adjInfo.nsAdjInfo.push_back({10, 1, 0});
    adjInfo.nsAdjInfo.push_back({20, 2, 0});
    adjInfo.nsAdjInfo.push_back({30, 3, 0});

    HcclResult ret = inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 1U);
    const auto& entry = inst_->hcclNslbDpAlgorithmInfo_[0];
    EXPECT_EQ(entry.taskId, DUMMY_TASK_ID);
    EXPECT_STREQ(entry.commDesc, kTestCommDesc);
    EXPECT_EQ(entry.srcLocalRankId, 1U);
    EXPECT_EQ(entry.rootRank, 0U);
    EXPECT_EQ(entry.oper, NSLBDP_CMD_ALLREDUCE);
    EXPECT_EQ(entry.algorithm, NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(entry.dstRankNum, 3U);
    EXPECT_EQ(entry.AdjInfo.size(), 3U);
    EXPECT_EQ(entry.AdjInfo[0].dstLocalRankId, 10U);
}

TEST_F(NslbDpStFlowTest, GetAlgAdjacencyTable_Deduplication)
{
    AdjInfo adjInfo = {};
    adjInfo.dstRankNum = 2;
    adjInfo.nsAdjInfo.push_back({10, 1, 0});
    adjInfo.nsAdjInfo.push_back({20, 2, 0});

    // 第一次：应该添加
    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLGATHER, 2U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    ASSERT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 1U);

    // 第二次：参数完全相同 → 去重命中，不新增
    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLGATHER, 2U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 1U);

    // 第三次：srcLocalRankId 不同 → 不算重复，新增
    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLGATHER, 9U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 2U);
}

TEST_F(NslbDpStFlowTest, GetAlgAdjacencyTable_EarlyReturns)
{
    // taskId == 0 → 门控挡住
    inst_->SetGlobalCommTaskId(0);
    AdjInfo adjInfo = {};
    adjInfo.dstRankNum = 1;
    adjInfo.nsAdjInfo.push_back({10, 1, 0});
    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 0U);

    // 恢复 taskId，用不支持的 opType → CheckSupportOptype 挡住
    inst_->SetGlobalCommTaskId(DUMMY_TASK_ID);
    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_GATHER, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 0U);

    // commDesc 不存在于 commConfig → InitAlgInfoCommDesc 返回 false
    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string("nonexistent_group"), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 0U);
}

// ============================================================
// 17. GenerateOpAndAdjTable：ST 全路径
// ============================================================
TEST_F(NslbDpStFlowTest, GenerateOpAndAdjTable_FullPath)
{
    HcclResult ret = inst_->GenerateOpAndAdjTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 0U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), 100ULL, 8U);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_EQ(inst_->hcclNslbDpOperatorVal_.size(), 1U);
    const auto& entry = inst_->hcclNslbDpOperatorVal_[0];
    EXPECT_EQ(entry.taskId, DUMMY_TASK_ID);
    EXPECT_STREQ(entry.commDesc, kTestCommDesc);
    EXPECT_EQ(entry.oper, NSLBDP_CMD_ALLREDUCE);
    EXPECT_EQ(entry.algorithm, NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(entry.rootRank, 0U);
    EXPECT_EQ(entry.sedFlag, 0U);
    EXPECT_NE(entry.trafficCnt, 0ULL);
}

TEST_F(NslbDpStFlowTest, GenerateOpAndAdjTable_EarlyReturns)
{
    // taskId == 0 → 提前返回
    inst_->SetGlobalCommTaskId(0);
    inst_->GenerateOpAndAdjTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 0U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), 100ULL, 8U);
    EXPECT_EQ(inst_->hcclNslbDpOperatorVal_.size(), 0U);

    inst_->SetGlobalCommTaskId(DUMMY_TASK_ID);

    // rootRank != 0 → 提前返回
    inst_->GenerateOpAndAdjTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), 100ULL, 8U);
    EXPECT_EQ(inst_->hcclNslbDpOperatorVal_.size(), 0U);

    // commDesc 不存在 → CheckCommDescExit 提前返回
    inst_->GenerateOpAndAdjTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 0U, 0U, NSLB_ALGO_TYPE_RING, std::string("no_such_group"), 100ULL, 8U);
    EXPECT_EQ(inst_->hcclNslbDpOperatorVal_.size(), 0U);
}

// ============================================================
// 18. SendOpAndAdjTable / SendAlgorithmInfoTable（需 netco flag）
// ============================================================
TEST_F(NslbDpStFlowTest, SendOpAndAdjTable_IteratesAndSends)
{
    // 先注入一条 operator 记录
    NslbDpOperatorInfo op = {};
    op.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(op.commDesc, COMM_DESC_MAX_LENGTH, kTestCommDesc, COMM_DESC_MAX_LENGTH - 1);
    op.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    op.oper = NSLBDP_CMD_ALLREDUCE;
    op.algorithm = NSLB_ALGO_TYPE_RING;
    op.rootRank = 0;
    op.trafficCnt = 42;
    inst_->hcclNslbDpOperatorVal_.push_back(op);

    // 设置 netco flag 使能发送路径（序列化会执行，H2DTlvRequest 可能返回错误但不崩溃）
    inst_->nslbdpIsInitNetCo_ = true;

    HcclResult ret = inst_->SendOpAndAdjTable();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // sedFlag 应为 1（已标记为已发送）
    EXPECT_EQ(inst_->hcclNslbDpOperatorVal_[0].sedFlag, 1U);
}

TEST_F(NslbDpStFlowTest, SendAlgorithmInfoTable_IteratesAndSends)
{
    // 先注入一条 algorithm 记录
    NslbDpAlgorithmInfo ai = {};
    ai.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(ai.commDesc, COMM_DESC_MAX_LENGTH, kTestCommDesc, COMM_DESC_MAX_LENGTH - 1);
    ai.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    ai.srcLocalRankId = 1;
    ai.oper = NSLBDP_CMD_ALLREDUCE;
    ai.algorithm = NSLB_ALGO_TYPE_RING;
    ai.rootRank = 0;
    ai.dstRankNum = 2;
    ai.AdjInfo.push_back({10, 1, 0});
    ai.AdjInfo.push_back({20, 2, 0});
    inst_->hcclNslbDpAlgorithmInfo_.push_back(ai);

    inst_->nslbdpIsInitNetCo_ = true;

    HcclResult ret = inst_->SendAlgorithmInfoTable();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // sedFlag 应为 1（已标记为已发送）
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_[0].sedFlag, 1U);
}

// ============================================================
// 19. 序列化函数直接覆盖（private 成员，-fno-access-control 可见）
// ============================================================
TEST_F(NslbDpStFlowTest, SerializeTLV_TableOpAndAdj_ReturnsNonEmpty)
{
    NslbDpOperatorInfo info = {};
    info.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(info.commDesc, COMM_DESC_MAX_LENGTH, "serialize_test", COMM_DESC_MAX_LENGTH - 1);
    info.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    info.oper = NSLBDP_CMD_ALLREDUCE;
    info.algorithm = NSLB_ALGO_TYPE_RING;
    info.rootRank = 0;
    info.trafficCnt = 99;
    info.l4SPortId = 8080;

    std::vector<uint8_t> result = inst_->serializeTLV_TableOpAndAdj(info);
    EXPECT_FALSE(result.empty());
    // 至少包含 taskId(8) + commDesc(128) + commInitTime(8) + 其他字段
    EXPECT_GT(result.size(), 150U);
}

TEST_F(NslbDpStFlowTest, SerializeTLV_TableAlgorithmInfo_ReturnsNonEmpty)
{
    NslbDpAlgorithmTlv info = {};
    info.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(info.commDesc, COMM_DESC_MAX_LENGTH, "seri_algo_test", COMM_DESC_MAX_LENGTH - 1);
    info.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    info.srcLocalRankId = 3;
    info.oper = NSLBDP_CMD_ALLGATHER;
    info.algorithm = NSLB_ALGO_TYPE_PIPELINE;
    info.rootRank = 1;
    info.dstRankNum = 2;
    info.AdjInfo.push_back({50, 1, 0});

    std::vector<uint8_t> result = inst_->serializeTLV_TableAlgorithmInfo(info);
    EXPECT_FALSE(result.empty());
}
