#include <iostream>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "rank_graph_interface.h"
#include "rank_graph_v2.h"
#include "hcomm_c_adpt.h"
#include "channel_process.h"
#include "base_config.h"
#define private public
#include "my_rank.h"
#undef private
#include "hccl_comm_pub.h"
#include "hccl/hccl_comm.h"
#include "llt_hccl_stub_rank_graph.h"

using namespace hccl;

class ExchangeInfoMgrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ExchangeInfoMgrTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ExchangeInfoMgrTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in ExchangeInfoMgrTest SetUP" << std::endl;
        rankIpPortMap = std::make_shared<std::unordered_map<u32, std::unordered_map<Hccl::IpAddress, u32>>>();
        (*rankIpPortMap)[0][Hccl::IpAddress("1.0.0.0")] = 16666;
        (*rankIpPortMap)[1][Hccl::IpAddress("2.0.0.0")] = 16666;
        (*rankIpPortMap)[2][Hccl::IpAddress("0.0.0.0")] = 16666;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ExchangeInfoMgrTest TearDown" << std::endl;
    }

    Hccl::RankIpPortMapPtr rankIpPortMap;
};

void InitCollComm(std::shared_ptr<hccl::hcclComm> hcclCommPtr)
{
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    void* commV2 = (void*)0x2000;
    uint32_t rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
}

TEST_F(ExchangeInfoMgrTest, Ut_WaitAllAsyncComplete_When_AllOk_Expect_Success)
{
    // mock Socket::GetAsyncStatus返回OK
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus)
        .stubs()
        .will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void *rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    // 构造socket列表（指针值仅用于mock匹配，不实际调用）
    std::vector<Hccl::Socket*> sockets = {(Hccl::Socket*)0x1, (Hccl::Socket*)0x2};
    std::vector<u32> remoteRanks = {1, 2};
    ExchangeInfoMgr exchangeInfoMgr;
    HcclResult ret = exchangeInfoMgr.WaitAllAsyncComplete(sockets, remoteRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_NewRankConsistent_Expect_Success)
{
    HcclResult ret = HCCL_SUCCESS;
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>();;
    InitCollComm(hcclCommPtr);
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    hccl::MyRank* myRank = collComm->GetMyRank();
    CollCommConfigConsistency &collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};
    ret = collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // mock Socket异步接口：GetAsyncStatus返回OK
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus)
        .stubs()
        .will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));
    // mock SendAsync/RecvAsync成功
    MOCKER_CPP(&Hccl::Socket::SendAsync)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::RecvAsync)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    // mock 超时配置
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut)
        .stubs()
        .will(returnValue((s32)30));

    HcclChannelDesc channelDescs[1];
    channelDescs[0].remoteRank = 1;
    HcommChannelDesc hcommDesc;
    hcommDesc.socket = (HcommSocket)0x1;  // 非空socket
    hcommDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
    std::vector<HcommChannelDesc> hcommDescVec;
    hcommDescVec.push_back(hcommDesc);
    ExchangeInfoMgr exchangeInfoMgr;
    ret = exchangeInfoMgr.BatchExchangeAndCheckConsistency(channelDescs, hcommDescVec, 1, collCommConfigConsistency, "test_tag");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
