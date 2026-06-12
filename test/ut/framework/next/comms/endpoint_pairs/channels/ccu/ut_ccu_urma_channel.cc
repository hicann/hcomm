#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#include "next/comms/endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#undef private

using namespace hcomm;

class CcuUrmaChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CcuUrmaChannelTest, Ut_Clean_When_ImplIsNull_Expect_HCCL_E_PTR) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);
}

// Minimal test-only connection deriving from CcuConnection to avoid heavy Init()
class TestCcuConnection : public CcuConnection {
public:
    TestCcuConnection(const CommAddr &locAddr, const CommAddr &rmtAddr,
        const CcuChannelInfo &channelInfo, const std::vector<CcuJetty *> &ccuJettys)
        : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys, 4U) {}
    // Do not call Init(); use default base behavior for Clean()
};

TEST_F(CcuUrmaChannelTest, Ut_Clean_When_ImplIsPresent_Expect_HCCL_SUCCESS) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    // Prepare minimal safe objects to construct a CcuTransport without heavy Init
    CommAddr locAddr{};
    CommAddr rmtAddr{};
    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty *> jettys{};

    // Create a test connection (does not call Init)
    std::unique_ptr<CcuConnection> conn(new TestCcuConnection(locAddr, rmtAddr, channelInfo, jettys));

    // Fake socket pointer (not dereferenced by Clean())
    Hccl::Socket *fakeSocket = reinterpret_cast<Hccl::Socket *>(0x1);

    // Prepare a simple buffer info
    CcuTransport::CclBufferInfo bufInfo(0x1000, 0x100, 1, 1);

    // Construct transport instance (constructor does not call Init)
    std::unique_ptr<CcuTransport> transport(new CcuTransport(fakeSocket, std::move(conn), bufInfo));

    // Inject into channel (we used #define private public when included)
    ch.impl_ = std::move(transport);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuUrmaChannelTest, Ut_Resume_When_Called_Expect_HCCL_SUCCESS) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    auto ret = ch.Resume();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_DfxInfo_TEST) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(1);
    CcuUrmaChannel ch(ep, desc);

    // 1. 构造依赖
    CommAddr locAddr{}, rmtAddr{};
    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty*> jettys{};

    auto conn = std::make_unique<CcuConnection>(locAddr, rmtAddr, channelInfo, jettys, 4U);
    auto fakeSocket = reinterpret_cast<Hccl::Socket*>(1);
    CcuTransport::CclBufferInfo bufInfo(0x1000, 0x100, 1, 1);

    // 2. 创建 transport
    auto transport = std::make_unique<CcuTransport>(fakeSocket, std::move(conn), bufInfo);
    ch.impl_ = std::move(transport);

    // 3. 从 ch.impl_ 操作，不要再用旧的 transport 指针！
    ch.impl_->transStatus_ = CcuTransport::TransStatus::SOCKET_TIMEOUT;
    auto ret = ch.GetStatus();
    EXPECT_NE(ret, ChannelStatus::INIT);

    ch.impl_->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    ret = ch.GetStatus();
    EXPECT_NE(ret, ChannelStatus::INIT);

    // 4. Mock 1：成功
    ch.impl_->transStatus_ = CcuTransport::TransStatus::READY;
    MOCKER_CPP(&CcuConnection::Describe, HcclResult (CcuConnection::*)(std::string&))
        .stubs()
        .with(any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::READY);
    GlobalMockObject::verify();
    GlobalMockObject::reset(); // <-- 必须重置 Mock！

    // 5. Mock 2：失败
    ch.impl_->transStatus_ = CcuTransport::TransStatus::READY;
    ch.isFirstPrintChannelInfo_ = true; // 需要重置为true才能触发Describe的调用
    MOCKER_CPP(&CcuConnection::Describe, HcclResult (CcuConnection::*)(std::string&))
        .stubs()
        .with(any())
        .will(returnValue(HcclResult::HCCL_E_PARA));
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::FAILED);
    GlobalMockObject::verify();
    GlobalMockObject::reset();
}
