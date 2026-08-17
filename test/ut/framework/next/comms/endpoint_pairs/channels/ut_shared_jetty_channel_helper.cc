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
#include "shared_jetty_channel_helper.h"
#include "endpoint.h"
#include "dev_ub_connection.h"
#include "shared_jetty_connection_adapter.h"
#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

namespace {
class StubEndpointForHelper : public Endpoint {
public:
    StubEndpointForHelper() : Endpoint(MakeDesc()) { ctxHandle_ = reinterpret_cast<void*>(0x1); }

    ~StubEndpointForHelper() override
    {
        if (sharedJettyCtx_.valid && sharedJettyCtx_.refCount > 0) {
            (void)ReleaseSharedJetty();
        }
    }

    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult ServerSocketListen(const uint32_t) override { return HCCL_SUCCESS; }
    HcclResult RegisterMemory(HcommMem, const char*, void**) override { return HCCL_SUCCESS; }
    HcclResult UnregisterMemory(void*) override { return HCCL_SUCCESS; }
    HcclResult MemoryExport(void*, void**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult MemoryImport(const void*, uint32_t, HcommMem*) override { return HCCL_SUCCESS; }
    HcclResult MemoryUnimport(const void*, uint32_t) override { return HCCL_SUCCESS; }
    HcclResult GetAllMemHandles(void**, uint32_t*) override { return HCCL_SUCCESS; }

private:
    static EndpointDesc MakeDesc()
    {
        EndpointDesc desc{};
        Hccl::IpAddress localIp("127.0.0.1");
        desc.protocol = COMM_PROTOCOL_UBC_CTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        return desc;
    }
};

std::unique_ptr<DevUbConnection> MakeTestConnection()
{
    IpAddress locIp("1.0.0.1");
    IpAddress rmtIp("2.0.0.2");
    return std::make_unique<DevUbConnection>(nullptr, locIp, rmtIp, OpMode::OPBASE);
}
} // namespace

class SharedJettyChannelHelperTest : public testing::Test {
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

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_NullEndpoint_Expect_HCCL_E_PARA)
{
    auto conn = MakeTestConnection();
    auto factory = []() {
        return MakeTestConnection();
    };
    Endpoint::SharedJettyCtx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(nullptr, conn.get(), factory, outCtx), HCCL_E_PARA);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_NullConnection_Expect_HCCL_E_PARA)
{
    StubEndpointForHelper endpoint;
    auto factory = []() {
        return MakeTestConnection();
    };
    Endpoint::SharedJettyCtx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, nullptr, factory, outCtx), HCCL_E_PARA);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_FirstCreate_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn = MakeTestConnection();
    uint32_t factoryCallCount = 0;
    auto factory = [&factoryCallCount]() -> std::unique_ptr<DevUbConnection> {
        factoryCallCount++;
        return MakeTestConnection();
    };

    Endpoint::SharedJettyCtx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);
    EXPECT_TRUE(endpoint.sharedJettyCtx_.valid);
    EXPECT_EQ(endpoint.sharedJettyCtx_.refCount, 1U);
    EXPECT_TRUE(conn->isSharedJetty_);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_CacheHit_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn1 = MakeTestConnection();
    auto conn2 = MakeTestConnection();
    uint32_t factoryCallCount = 0;
    auto factory = [&factoryCallCount]() -> std::unique_ptr<DevUbConnection> {
        factoryCallCount++;
        return MakeTestConnection();
    };

    Endpoint::SharedJettyCtx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn1.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);

    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn2.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);
    EXPECT_EQ(endpoint.sharedJettyCtx_.refCount, 2U);
    EXPECT_TRUE(conn2->isSharedJetty_);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_ReleaseDecrementsRef_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn = MakeTestConnection();
    auto factory = []() {
        return MakeTestConnection();
    };

    Endpoint::SharedJettyCtx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.sharedJettyCtx_.refCount, 1U);

    EXPECT_EQ(endpoint.ReleaseSharedJetty(), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.sharedJettyCtx_.refCount, 0U);
    EXPECT_FALSE(endpoint.sharedJettyCtx_.valid);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_ReleaseAndReacquire_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn1 = MakeTestConnection();
    auto conn2 = MakeTestConnection();
    uint32_t factoryCallCount = 0;
    auto factory = [&factoryCallCount]() -> std::unique_ptr<DevUbConnection> {
        factoryCallCount++;
        return MakeTestConnection();
    };

    Endpoint::SharedJettyCtx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn1.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.ReleaseSharedJetty(), HCCL_SUCCESS);
    EXPECT_FALSE(endpoint.sharedJettyCtx_.valid);

    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn2.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 2U);
    EXPECT_TRUE(endpoint.sharedJettyCtx_.valid);
    EXPECT_EQ(endpoint.sharedJettyCtx_.refCount, 1U);
}
