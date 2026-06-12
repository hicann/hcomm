#include "gtest/gtest.h"
#include "../../hccl_api_base_test.h"
#include "hccl/hccl_res.h"
#include "hccl_common.h"
#include "independent_op_context_manager.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include "llt_hccl_stub_rank_graph.h"
#include <string>
#include "mockcpp/mockcpp.hpp"
#include "cluster_monitor.h"
#include "host/host_cpu_roce_channel.h"

#define private public

using namespace hccl;
using namespace hcomm;

HcclResult ProcessUbChannelDesc(const HcclChannelDesc &channelDesc, HcclChannelDesc &channelDescFinal,
    hcclComm *hcclComm);

class HcclChannelDescTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        const char *fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
        MOCKER(&HcclCommDfx::ReportKernel).stubs().will(returnValue(HCCL_SUCCESS));
        SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
protected: 
    void SetUpCommAndGraph(std::shared_ptr < hccl::hcclComm > &hcclCommPtr, 
        std::shared_ptr < Hccl::RankGraph > &rankGraphV2, void* &comm, HcclResult &ret) 
    {
        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

        bool isDeviceSide {
            false
        };
        MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
        MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        setenv("HCCL_RDMA_RETRY_CNT", "7", 1);
        setenv("HCCL_RDMA_TIMEOUT", "20", 1);
        setenv("HCCL_RDMA_TC", "120", 1);
        setenv("HCCL_RDMA_SL", "2", 1);
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        RankGraphStub rankGraphStub;
        rankGraphV2 = rankGraphStub.Create2PGraph();
        void* commV2 = (void*)0x2000;
        uint32_t rank = 1;
        HcclMem cclBuffer;
        cclBuffer.size = 1;
        cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
        cclBuffer.addr = (void*)0x1000;
        char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
        hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);
        HcclCommConfig config;
        UtInitHcclCommConfig(config);
        config.hcclOpExpansionMode = 1; // 非CCU模式，避免拉起CCU平台层
        config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
        config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level 
        unsetenv("HCCL_DFS_CONFIG");      
        ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
        CollComm* collComm = hcclCommPtr->GetCollComm();
        comm = static_cast<HcclComm>(hcclCommPtr.get());
    }

    void GetChannelDesc(std::vector<HcclChannelDesc> &channelDesc)
    {
        HcclChannelDescInit(channelDesc.data(), 1);
        channelDesc[0].remoteRank = 2;
        channelDesc[0].channelProtocol = CommProtocol::COMM_PROTOCOL_ROCE;
        channelDesc[0].notifyNum = 50;
        channelDesc[0].roceAttr.queueNum = 3;
        channelDesc[0].roceAttr.retryCnt = 3;
        channelDesc[0].roceAttr.retryInterval = 20;
        channelDesc[0].roceAttr.tc = 120;
        channelDesc[0].roceAttr.sl = 3;
    }
private:
    std::shared_ptr<hccl::hcclComm>hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph>rankGraphV2;
    void* comm;
    HcclResult ret;
};

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_IsCommunicatorV2_Is_True_RetrunHCCLSUCCESS)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    MOCKER(&MyRank::CreateChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommDpuChannelRegisterDfx).stubs().will(returnValue(0));
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_TcIsInvaild_ReturnHCCLEPARA)
{
    hcclCommPtr->collComm_->config_.trafficClass_ = 10; // 单独赋非法值
    comm = static_cast<HcclComm>(hcclCommPtr.get());    // 重新给comm

    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_SlIsInvaild_ReturnHCCLEPARA)
{
    hcclCommPtr->collComm_->config_.serviceLevel_ = 10; // 单独赋非法值
    comm = static_cast<HcclComm>(hcclCommPtr.get());    // 重新给comm

    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_RetryIntervalIsInvaild_ReturnHCCLEPARA)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    channelDesc[0].roceAttr.retryInterval = 30; // 单独赋非法值

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_RetryCntIsInvaild_ReturnHCCLEPARA)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    channelDesc[0].roceAttr.retryCnt = 10; // 单独赋非法值
    MOCKER(&MyRank::BatchCreateSockets).stubs().will(returnValue(HCCL_SUCCESS));
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_Notifynum_Exceeds_Return_Error)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    channelDesc[0].notifyNum = 65;
    MOCKER(&hcomm::ClusterMonitor::RegisterToClusterMonitor).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_BuildConnection_Fails_Return_Error)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    MOCKER_CPP(&MyRank::BatchCreateSockets).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommEndpointCreate).stubs().will(returnValue(0));
    // Mock BuildConnection 失败
    MOCKER(&HostCpuRoceChannel::BuildConnection).stubs().will(returnValue(HCCL_E_NETWORK));
    
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_IbvPostRecv_Fails_Return_Error)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    MOCKER_CPP(&MyRank::BatchCreateSockets).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommEndpointCreate).stubs().will(returnValue(0));
    // Mock BuildConnection 成功
    MOCKER(&HostCpuRoceChannel::BuildConnection).stubs().will(returnValue(HCCL_SUCCESS));
    // Mock IbvPostRecv 失败
    MOCKER(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_E_INTERNAL));
    
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_WrongProtocol_Expect_E_PARA)
{
    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_ROCE;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_Hccs_Expect_E_PARA)
{
    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_HCCS;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_UbcCtp_QosUnset_UsesCommHcclQos)
{
    hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(5u);
    comm = static_cast<HcclComm>(hcclCommPtr.get());

    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_CTP;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_UbcTp_QosUnset_UsesUbQosDefault)
{
    hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(HCCL_COMM_QOS_CONFIG_NOT_SET);
    comm = static_cast<HcclComm>(hcclCommPtr.get());

    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_TP;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_UbcCtp_Valid_Expect_Success)
{
    hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(1u);
    comm = static_cast<HcclComm>(hcclCommPtr.get());

    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_CTP;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_Uboe_QosUnset_UsesCommHcclQos)
{
    hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(3u);
    comm = static_cast<HcclComm>(hcclCommPtr.get());

    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBOE;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}