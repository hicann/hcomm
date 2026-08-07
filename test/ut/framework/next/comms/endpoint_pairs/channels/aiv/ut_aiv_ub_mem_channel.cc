#include "gtest/gtest.h"
#include <cstring>

#include "aiv_ub_mem_channel.h"

using namespace hcomm;

class AivUbMemChannelTest : public testing::Test {
protected:
    HcommChannelDesc MakeDefaultDesc()
    {
        HcommChannelDesc desc;
        (void)std::memset(&desc, 0, sizeof(desc));
        return desc;
    }

    AivUbMemChannel* CreateChannel()
    {
        EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
        HcommChannelDesc desc = MakeDefaultDesc();
        return new AivUbMemChannel(ep, desc);
    }

    void DestroyChannel(AivUbMemChannel* ch) { delete ch; }
};

// Clean: 调用应返回 HCCL_SUCCESS，无论调用次数
TEST_F(AivUbMemChannelTest, Ut_CleanWhenCalledExpectHCCLSuccess)
{
    AivUbMemChannel* ch = CreateChannel();
    EXPECT_EQ(ch->Clean(), HCCL_SUCCESS);
    // 重复调用覆盖相同行为
    EXPECT_EQ(ch->Clean(), HCCL_SUCCESS);
    DestroyChannel(ch);
}

// Resume: 调用应返回 HCCL_SUCCESS，无论调用次数
TEST_F(AivUbMemChannelTest, Ut_ResumeWhenCalledExpectHCCLSuccess)
{
    AivUbMemChannel* ch = CreateChannel();
    EXPECT_EQ(ch->Resume(), HCCL_SUCCESS);
    // 重复调用覆盖相同行为
    EXPECT_EQ(ch->Resume(), HCCL_SUCCESS);
    DestroyChannel(ch);
}
