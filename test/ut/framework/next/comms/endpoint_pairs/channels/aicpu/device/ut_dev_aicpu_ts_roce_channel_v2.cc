/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "dev_aicpu_ts_roce_channel_v2.h"
#include "roce_transport_lite_impl.h"
#include "rma_buffer_lite.h"
#include "rmt_rma_buffer_lite.h"
#include "notify_lite.h"
#include "binary_stream.h"
#include "rdma_conn_lite_v2.h"

#define private public

using namespace Hccl;

static const u32 ROCE_TYPE = 1;

static std::vector<char> BuildSingleRmaBufferUniqueId(u64 addr, u64 size, u32 key)
{
    BinaryStream bs;
    bs << addr;
    bs << size;
    bs << key;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildLocNotifyUniqueIds(u32 notifyNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < notifyNum; i++) {
        u32 notifyId = i + 1;
        u32 devPhyId = i + 10;
        result.insert(
            result.end(), reinterpret_cast<char*>(&notifyId), reinterpret_cast<char*>(&notifyId) + sizeof(notifyId));
        result.insert(
            result.end(), reinterpret_cast<char*>(&devPhyId), reinterpret_cast<char*>(&devPhyId) + sizeof(devPhyId));
    }
    return result;
}

static std::vector<char> BuildRmtNotifyUniqueIds(u32 notifyNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < notifyNum; i++) {
        u64 addr = 0x2000 + i * 0x100;
        u64 size = 0x100;
        u32 rkey = 0x200 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&rkey), reinterpret_cast<char*>(&rkey) + sizeof(rkey));
    }
    return result;
}

static std::vector<char> BuildNotifyValueBufferUniqueId() { return BuildSingleRmaBufferUniqueId(0x3000, 0x200, 0x300); }

static std::vector<char> BuildLocBufferUniqueIds(u32 bufferNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < bufferNum; i++) {
        u64 addr = 0x4000 + i * 0x200;
        u64 size = 0x200;
        u32 lkey = 0x400 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&lkey), reinterpret_cast<char*>(&lkey) + sizeof(lkey));
    }
    return result;
}

static std::vector<char> BuildRmtBufferUniqueIds(u32 bufferNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < bufferNum; i++) {
        u64 addr = 0x5000 + i * 0x200;
        u64 size = 0x200;
        u32 rkey = 0x600 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&rkey), reinterpret_cast<char*>(&rkey) + sizeof(rkey));
    }
    return result;
}

static std::vector<char> BuildSqUniqueId(const RdmaSqContextLite& sqCtx)
{
    BinaryStream bs;
    bs << sqCtx.qpn;
    bs << sqCtx.sqVa;
    bs << sqCtx.wqeSize;
    bs << sqCtx.depth;
    bs << sqCtx.headAddr;
    bs << sqCtx.tailAddr;
    bs << sqCtx.dbHwVa;
    bs << sqCtx.dbSwVa;
    bs << sqCtx.sl;
    bs << sqCtx.dbVendorSpecified;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildCqUniqueId(const RdmaCqContextLite& cqCtx)
{
    BinaryStream bs;
    bs << cqCtx.cqn;
    bs << cqCtx.cqVa;
    bs << cqCtx.cqeSize;
    bs << cqCtx.cqDepth;
    bs << cqCtx.headAddr;
    bs << cqCtx.tailAddr;
    bs << cqCtx.dbSwVa;
    bs << cqCtx.dbVendorSpecified;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char>
BuildSingleConnUniqueId(u32 dmaMode, const RdmaSqContextLite& sqCtx, const RdmaCqContextLite& cqCtx)
{
    BinaryStream bs;
    bs << dmaMode;

    std::vector<char> sqUniqueId = BuildSqUniqueId(sqCtx);
    bs << sqUniqueId;

    std::vector<char> cqUniqueId = BuildCqUniqueId(cqCtx);
    bs << cqUniqueId;

    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildConnUniqueIds(u32 connNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < connNum; i++) {
        RdmaSqContextLite sq{};
        sq.qpn = 1 + i;
        sq.sqVa = 0x10000;
        sq.wqeSize = 64;
        sq.depth = 128;
        sq.headAddr = 0x20000;
        sq.tailAddr = 0x20008;
        sq.sl = 0;
        sq.dbHwVa = 0x30000;
        sq.dbSwVa = 0x70000;
        sq.dbVendorSpecified = 0;
        sq.dbVendorSpecified
            |= (static_cast<uint64_t>(static_cast<uint8_t>(3) & UB_DB_VENDOR_FIELD_MASK)
                << UB_DB_VENDOR_MTUSHIFT_SHIFT);
        sq.dbVendorSpecified
            |= (static_cast<uint64_t>(static_cast<uint8_t>(0x7) & UB_DB_VENDOR_FIELD_MASK) << UB_DB_VENDOR_COS_SHIFT);

        RdmaCqContextLite cq{};
        cq.cqn = 2 + i;
        cq.cqVa = 0x40000;
        cq.cqeSize = 32;
        cq.cqDepth = 256;
        cq.headAddr = 0x50000;
        cq.tailAddr = 0x50008;
        cq.dbSwVa = 0x60000;
        cq.dbVendorSpecified = 0;

        std::vector<char> singleConn = BuildSingleConnUniqueId(0, sq, cq);
        result.insert(result.end(), singleConn.begin(), singleConn.end());
    }
    return result;
}

static std::vector<char> BuildChannelUniqueId(u32 notifyNum, u32 bufferNum, u32 connNum)
{
    BinaryStream bs;
    u32 type = ROCE_TYPE;
    bs << type;
    bs << notifyNum;
    bs << bufferNum;
    bs << connNum;

    std::vector<char> locNotifyIds = BuildLocNotifyUniqueIds(notifyNum);
    bs << locNotifyIds;

    std::vector<char> rmtNotifyIds = BuildRmtNotifyUniqueIds(notifyNum);
    bs << rmtNotifyIds;

    std::vector<char> notifyValueId = BuildNotifyValueBufferUniqueId();
    bs << notifyValueId;

    std::vector<char> locBufferIds = BuildLocBufferUniqueIds(bufferNum);
    bs << locBufferIds;

    std::vector<char> rmtBufferIds = BuildRmtBufferUniqueIds(bufferNum);
    bs << rmtBufferIds;

    std::vector<char> connIds = BuildConnUniqueIds(connNum);
    bs << connIds;

    std::vector<char> result;
    bs.Dump(result);
    return result;
}

class DevAicpuTsRoceChannelV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DevAicpuTsRoceChannelV2Test tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "DevAicpuTsRoceChannelV2Test tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in DevAicpuTsRoceChannelV2Test SetUP" << std::endl;
        uniqueId_ = BuildChannelUniqueId(NOTIFY_NUM, BUFFER_NUM, CONN_NUM);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DevAicpuTsRoceChannelV2Test TearDown" << std::endl;
    }

    u32 NOTIFY_NUM = 2;
    u32 BUFFER_NUM = 1;
    u32 CONN_NUM = 1;
    std::vector<char> uniqueId_;
};

TEST_F(DevAicpuTsRoceChannelV2Test, Ut_When_Construct_Expect_Success)
{
    std::cout << "Start Ut_When_Construct_Expect_Success" << std::endl;

    DevAicpuTsRoceChannelV2 channel(uniqueId_);

    std::cout << "End Ut_When_Construct_Expect_Success" << std::endl;
}

TEST_F(DevAicpuTsRoceChannelV2Test, Ut_When_Describe_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_Describe_Expect_NotEmpty" << std::endl;

    DevAicpuTsRoceChannelV2 channel(uniqueId_);

    std::string desc = channel.Describe();
    EXPECT_FALSE(desc.empty());

    std::cout << "End Ut_When_Describe_Expect_NotEmpty" << std::endl;
}
