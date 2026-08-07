/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "ccu_transport_.h"
#include "binary_stream.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

class CclBufferInfoTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CclBufferInfoTest, Ut_PackUnpack_RoundTrip_Expect_SameValues)
{
    CcuTransport::CclBufferInfo original(0x12345678ULL, 1024U, 100U, 200U);
    Hccl::BinaryStream stream;
    original.Pack(stream);

    CcuTransport::CclBufferInfo restored;
    restored.Unpack(stream);
    EXPECT_EQ(restored.addr, 0x12345678ULL);
    EXPECT_EQ(restored.size, 1024U);
    EXPECT_EQ(restored.tokenId, 100U);
    EXPECT_EQ(restored.tokenValue, 200U);
}

TEST_F(CclBufferInfoTest, Ut_PackUnpack_WithMemInfo_Expect_SameValues)
{
    CcuTransport::CclBufferInfo original(0xDEADBEEFULL, 4096U, 10U, 20U);
    std::array<char, HCCL_RES_TAG_MAX_LEN> memInfo{};
    const char* tag = "TestBuffer";
    memcpy(memInfo.data(), tag, strlen(tag));
    original.memInfo = memInfo;

    Hccl::BinaryStream stream;
    original.Pack(stream);

    CcuTransport::CclBufferInfo restored;
    restored.Unpack(stream);
    EXPECT_EQ(restored.addr, 0xDEADBEEFULL);
    EXPECT_EQ(restored.size, 4096U);
    std::string restoredTag(restored.memInfo.data(), strnlen(restored.memInfo.data(), HCCL_RES_TAG_MAX_LEN));
    EXPECT_EQ(restoredTag, "TestBuffer");
}

TEST_F(CclBufferInfoTest, Ut_DefaultConstructor_Expect_ZeroValues)
{
    CcuTransport::CclBufferInfo info;
    EXPECT_EQ(info.addr, 0ULL);
    EXPECT_EQ(info.size, 0U);
    EXPECT_EQ(info.tokenId, 0U);
    EXPECT_EQ(info.tokenValue, 0U);
}

TEST_F(CclBufferInfoTest, Ut_ConstructorWithParams_Expect_CorrectValues)
{
    CcuTransport::CclBufferInfo info(0x1000ULL, 512U, 1U, 2U);
    EXPECT_EQ(info.addr, 0x1000ULL);
    EXPECT_EQ(info.size, 512U);
    EXPECT_EQ(info.tokenId, 1U);
    EXPECT_EQ(info.tokenValue, 2U);
}

class CcuTransportTest : public testing::Test {
protected:
    std::unique_ptr<CcuTransport> transport;
    void SetUp() override
    {
        CcuTransport::CclBufferInfo bufInfo(0x1000ULL, 1024U, 1U, 2U);
        transport = std::make_unique<CcuTransport>(nullptr, nullptr, bufInfo);
    }
    void TearDown() override {}
};

TEST_F(CcuTransportTest, Ut_GetDieId_When_Set_Expect_CorrectValue)
{
    transport->dieId_ = 1;
    EXPECT_EQ(transport->GetDieId(), 1U);
}

TEST_F(CcuTransportTest, Ut_GetLocCkeByIndex_When_Empty_Expect_ReturnHCCL_E_PARA)
{
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetLocCkeByIndex(0, ckeId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetLocCkeByIndex_When_IndexOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    transport->locRes_.ckes = {10, 20, 30};
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetLocCkeByIndex(5, ckeId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetLocCkeByIndex_When_Normal_Expect_ReturnSuccess)
{
    transport->locRes_.ckes = {10, 20, 30};
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetLocCkeByIndex(1, ckeId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ckeId, 20U);
}

TEST_F(CcuTransportTest, Ut_GetLocXnByIndex_When_Empty_Expect_ReturnHCCL_E_PARA)
{
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetLocXnByIndex(0, xnId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetLocXnByIndex_When_Normal_Expect_ReturnSuccess)
{
    transport->locRes_.xns = {100, 200};
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetLocXnByIndex(0, xnId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(xnId, 100U);
}

TEST_F(CcuTransportTest, Ut_GetLocXnByIndex_When_IndexOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    transport->locRes_.xns = {100, 200};
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetLocXnByIndex(5, xnId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetRmtCkeByIndex_When_Empty_Expect_ReturnHCCL_E_PARA)
{
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetRmtCkeByIndex(0, ckeId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetRmtCkeByIndex_When_Normal_Expect_ReturnSuccess)
{
    transport->rmtRes_.ckes = {50, 60};
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetRmtCkeByIndex(0, ckeId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ckeId, 50U);
}

TEST_F(CcuTransportTest, Ut_GetRmtXnByIndex_When_Empty_Expect_ReturnHCCL_E_PARA)
{
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetRmtXnByIndex(0, xnId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetRmtXnByIndex_When_Normal_Expect_ReturnSuccess)
{
    transport->rmtRes_.xns = {500, 600};
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetRmtXnByIndex(1, xnId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(xnId, 600U);
}

TEST_F(CcuTransportTest, Ut_GetRmtXnByIndex_When_IndexOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    transport->rmtRes_.xns = {500};
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetRmtXnByIndex(1, xnId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetLocBuffer_When_Normal_Expect_ReturnSuccess)
{
    CcuTransport::CclBufferInfo buf;
    EXPECT_EQ(transport->GetLocBuffer(buf, 0), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(buf.addr, 0x1000ULL);
}

TEST_F(CcuTransportTest, Ut_GetRmtBuffer_When_Normal_Expect_ReturnSuccess)
{
    transport->rmtHcclBufferInfo_ = CcuTransport::CclBufferInfo(0x2000ULL, 2048U, 3U, 4U);
    CcuTransport::CclBufferInfo buf;
    EXPECT_EQ(transport->GetRmtBuffer(buf, 0), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(buf.addr, 0x2000ULL);
}

TEST_F(CcuTransportTest, Ut_GetCkeNum_When_Normal_Expect_ReturnSuccess)
{
    transport->locRes_.ckes = {1, 2, 3, 4, 5};
    uint32_t ckeNum = 0;
    EXPECT_EQ(transport->GetCkeNum(ckeNum), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ckeNum, 5U);
}

TEST_F(CcuTransportTest, Ut_Describe_When_Normal_Expect_NonEmpty)
{
    transport->dieId_ = 1;
    transport->locRes_.ckes = {1, 2};
    transport->locRes_.xns = {3, 4};
    transport->rmtRes_.ckes = {5};
    transport->rmtRes_.xns = {6};
    std::string desc = transport->Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(CcuTransportTest, Ut_CheckFinish_When_MatchingMsg_Expect_ReturnSuccess)
{
    std::string finishMsg = "Transport exchange data ready!";
    transport->sendFinishMsg_ = std::vector<char>(finishMsg.begin(), finishMsg.end());
    transport->recvFinishMsg_ = std::vector<char>(finishMsg.begin(), finishMsg.end());
    EXPECT_EQ(transport->CheckFinish(), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuTransportTest, Ut_CheckFinish_When_MismatchMsg_Expect_ReturnHCCL_E_INTERNAL)
{
    std::string finishMsg = "Transport exchange data ready!";
    transport->sendFinishMsg_ = std::vector<char>(finishMsg.begin(), finishMsg.end());
    std::string wrongMsg(128, 'x');
    transport->recvFinishMsg_ = std::vector<char>(wrongMsg.begin(), wrongMsg.end());
    EXPECT_EQ(transport->CheckFinish(), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CcuTransportTest, Ut_ResUpdate_When_EmptyTags_Expect_ReturnSuccess)
{
    std::vector<std::string> tags;
    EXPECT_EQ(transport->ResUpdate(tags), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuTransportTest, Ut_ResUpdate_When_NewTags_Expect_InsertIntoCntXns)
{
    transport->transStatus_ = CcuTransport::TransStatus::READY;
    transport->dieId_ = 0;
    transport->devLogicId_ = 0;
    MOCKER(hcomm::CcuDevMgrImp::AllocWishCntXn).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    std::vector<std::string> tags = {"group1", "group2"};
    EXPECT_EQ(transport->ResUpdate(tags), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(transport->locRes_.cntXns.size(), 2U);
    EXPECT_EQ(transport->locRes_.cntXns.count("group1"), 1U);
    EXPECT_EQ(transport->locRes_.cntXns.count("group2"), 1U);
    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

TEST_F(CcuTransportTest, Ut_GetRmtWishCntXnAddr_When_NotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    uint64_t addr = 0;
    EXPECT_EQ(transport->GetRmtWishCntXnAddr("nonexistent", addr), HcclResult::HCCL_E_NOT_FOUND);
}

TEST_F(CcuTransportTest, Ut_AttributionDescribe_Expect_NonEmpty)
{
    CcuTransport::Attribution attr;
    attr.devicePhyId = 1;
    attr.handshakeMsg = {'a', 'b', 'c'};
    std::string desc = attr.Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(CcuTransportTest, Ut_SetHandshakeMsg_Then_GetLocalHandshakeMsg_Expect_Same)
{
    std::vector<char> msg = {'h', 'e', 'l', 'l', 'o'};
    transport->SetHandshakeMsg(msg);
    EXPECT_EQ(transport->GetLocalHandshakeMsg(), msg);
}

TEST_F(CcuTransportTest, Ut_GetRmtHandshakeMsg_Expect_InitiallyEmpty)
{
    transport->rmtHandshakeMsg_.clear();
    auto& msg = transport->GetRmtHandshakeMsg();
    EXPECT_TRUE(msg.empty());
}
