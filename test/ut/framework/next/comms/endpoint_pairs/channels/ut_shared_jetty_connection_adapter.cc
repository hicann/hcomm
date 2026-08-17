/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "shared_jetty_connection_adapter.h"
#include "endpoint.h"
#include "dev_ub_connection.h"
#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

namespace {
constexpr uint32_t DEFAULT_OPBASE_UB_SQ_DEPTH = 8192U;

std::unique_ptr<DevUbConnection> MakeTestConnection()
{
    IpAddress locIp("1.0.0.1");
    IpAddress rmtIp("2.0.0.2");
    return std::make_unique<DevUbConnection>(nullptr, locIp, rmtIp, OpMode::OPBASE);
}

Endpoint::SharedJettyCtx MakeTestCtx()
{
    Endpoint::SharedJettyCtx ctx;
    ctx.handle = 0x1234;
    ctx.handlePtr = reinterpret_cast<void*>(0x5678);
    ctx.jettyId = 100;
    ctx.sqBuffVa = 0xABCDEF00;
    ctx.dbAddr = 0x12345678;
    ctx.keySize = HRT_UB_QP_KEY_MAX_LEN;
    for (uint32_t i = 0; i < ctx.keySize; ++i) {
        ctx.localQpKey[i] = static_cast<uint8_t>(i);
    }
    ctx.sqDepth = 32;
    ctx.tpHandle = 0xDEADBEEF;
    return ctx;
}
} // namespace

class SharedJettyConnAdaptTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_NullConnection_Expect_HCCL_E_PARA)
{
    Endpoint::SharedJettyCtx ctx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    EXPECT_EQ(InjectSharedJettyToConn(nullptr, ctx, reinterpret_cast<void*>(0x1), releaseCb), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_Normal_Expect_Success)
{
    auto conn = MakeTestConnection();
    Endpoint::SharedJettyCtx ctx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    EXPECT_EQ(InjectSharedJettyToConn(conn.get(), ctx, reinterpret_cast<void*>(0x1), releaseCb), HCCL_SUCCESS);
    EXPECT_TRUE(conn->isSharedJetty_);
    EXPECT_EQ(conn->jettyHandle, static_cast<JettyHandle>(0x1234));
    EXPECT_EQ(conn->jettyId, 100U);
    EXPECT_EQ(conn->sqBuffVa, 0xABCDEF00ULL);
    EXPECT_EQ(conn->sqDepth, 32U);
    EXPECT_EQ(conn->tpInfo.tpHandle, 0xDEADBEEFULL);
    EXPECT_EQ(conn->endpointTag_, reinterpret_cast<void*>(0x1));
}

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_ZeroKeySize_Expect_Success)
{
    auto conn = MakeTestConnection();
    Endpoint::SharedJettyCtx ctx = MakeTestCtx();
    ctx.keySize = 0;
    auto releaseCb = [](void*) {};
    EXPECT_EQ(InjectSharedJettyToConn(conn.get(), ctx, reinterpret_cast<void*>(0x1), releaseCb), HCCL_SUCCESS);
    EXPECT_TRUE(conn->isSharedJetty_);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_NullConnection_Expect_HCCL_E_PARA)
{
    Endpoint::SharedJettyCtx ctx;
    EXPECT_EQ(ExtractJettyInfoFromConn(nullptr, ctx), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_Normal_Expect_Success)
{
    auto conn = MakeTestConnection();
    Endpoint::SharedJettyCtx injectCtx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    ASSERT_EQ(InjectSharedJettyToConn(conn.get(), injectCtx, reinterpret_cast<void*>(0x1), releaseCb), HCCL_SUCCESS);

    // 设置 JFC 相关字段，验证 ExtractJettyInfoFromConn 能正确提取
    conn->rdmaHandle = reinterpret_cast<void*>(0xABCD);
    conn->jfcHandle = 12345;

    Endpoint::SharedJettyCtx extractCtx;
    EXPECT_EQ(ExtractJettyInfoFromConn(conn.get(), extractCtx), HCCL_SUCCESS);
    EXPECT_EQ(extractCtx.handle, injectCtx.handle);
    EXPECT_EQ(extractCtx.jettyId, injectCtx.jettyId);
    EXPECT_EQ(extractCtx.sqBuffVa, injectCtx.sqBuffVa);
    EXPECT_EQ(extractCtx.dbAddr, injectCtx.dbAddr);
    EXPECT_EQ(extractCtx.keySize, injectCtx.keySize);
    EXPECT_EQ(extractCtx.sqDepth, injectCtx.sqDepth);
    EXPECT_EQ(extractCtx.tpHandle, injectCtx.tpHandle);
    EXPECT_EQ(extractCtx.rdmaHandle, reinterpret_cast<void*>(0xABCD));
    EXPECT_EQ(extractCtx.jfcHandle, 12345U);
    EXPECT_EQ(memcmp(extractCtx.localQpKey, injectCtx.localQpKey, HRT_UB_QP_KEY_MAX_LEN), 0);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_EmptyConnection_Expect_DefaultSqDepth)
{
    auto conn = MakeTestConnection();
    Endpoint::SharedJettyCtx ctx;
    EXPECT_EQ(ExtractJettyInfoFromConn(conn.get(), ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.handle, static_cast<JettyHandle>(0));
    EXPECT_EQ(ctx.jettyId, 0U);
    EXPECT_EQ(ctx.sqDepth, DEFAULT_OPBASE_UB_SQ_DEPTH);
    EXPECT_EQ(ctx.rdmaHandle, nullptr);
    EXPECT_EQ(ctx.jfcHandle, 0U);
}

TEST_F(SharedJettyConnAdaptTest, Ut_TransferOwnership_When_NullConnection_Expect_HCCL_E_PARA)
{
    EXPECT_EQ(TransferConnJettyOwnership(nullptr), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_TransferOwnership_When_Normal_Expect_Success)
{
    auto conn = MakeTestConnection();
    EXPECT_FALSE(conn->isSharedJetty_);
    EXPECT_EQ(TransferConnJettyOwnership(conn.get()), HCCL_SUCCESS);
    EXPECT_TRUE(conn->isSharedJetty_);
}

TEST_F(SharedJettyConnAdaptTest, Ut_RoundTrip_InjectExtractTransfer_Expect_Consistent)
{
    auto conn = MakeTestConnection();
    Endpoint::SharedJettyCtx injectCtx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    ASSERT_EQ(InjectSharedJettyToConn(conn.get(), injectCtx, reinterpret_cast<void*>(0x1), releaseCb), HCCL_SUCCESS);

    conn->rdmaHandle = reinterpret_cast<void*>(0xBEEF);
    conn->jfcHandle = 67890;

    ASSERT_EQ(TransferConnJettyOwnership(conn.get()), HCCL_SUCCESS);

    Endpoint::SharedJettyCtx extractCtx;
    ASSERT_EQ(ExtractJettyInfoFromConn(conn.get(), extractCtx), HCCL_SUCCESS);
    EXPECT_EQ(extractCtx.handle, injectCtx.handle);
    EXPECT_EQ(extractCtx.handlePtr, injectCtx.handlePtr);
    EXPECT_EQ(extractCtx.jettyId, injectCtx.jettyId);
    EXPECT_EQ(extractCtx.sqBuffVa, injectCtx.sqBuffVa);
    EXPECT_EQ(extractCtx.dbAddr, injectCtx.dbAddr);
    EXPECT_EQ(extractCtx.sqDepth, injectCtx.sqDepth);
    EXPECT_EQ(extractCtx.keySize, injectCtx.keySize);
    EXPECT_EQ(extractCtx.tpHandle, injectCtx.tpHandle);
    EXPECT_EQ(extractCtx.rdmaHandle, reinterpret_cast<void*>(0xBEEF));
    EXPECT_EQ(extractCtx.jfcHandle, 67890U);
}
