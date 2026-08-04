#include "gtest/gtest.h"
#include "../../hccl_api_base_test.h"
#include "hccl/hccl_res.h"
#include "hccl_common.h"
#include "config/env_config.h"
#include "independent_op_context_manager.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include "llt_hccl_stub_rank_graph.h"
#include <cstring>
#include <string>
#include "mockcpp/mockcpp.hpp"
#include "dfx/cluster_monitor/cluster_monitor.h"
#include "host/host_cpu_roce_channel.h"
#include "param_check_pub.h"
#include "hccl/hccl_types.h"
#include "hccp.h"
#include "my_rank.h"
#include "aiv_urma_channel.h"
#include "channel_process.h"

#define private public

using namespace hccl;
using namespace hcomm;

HcclResult ProcessUbChannelDesc(const HcclChannelDesc &channelDesc, const HcclChannelDesc &channelDescFinal,
    const hcclComm *hcclComm);
HcclResult ProcessHcclResPackReq(const HcclChannelDesc &channelDesc, HcclChannelDesc &channelDescFinal,
    hcclComm *hcclComm);

static int StubRaGetHccnCfgRoceQosDscp(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen)
{
    (void)info;
    (void)key;
    if (value == nullptr || valueLen == nullptr) {
        return -1;
    }
    const char *cfg = "0:33,1:20,4:50,6:70";
    const unsigned int len = static_cast<unsigned int>(std::strlen(cfg));
    if (*valueLen < len) {
        return -1;
    }
    (void)std::memcpy(value, cfg, len);
    *valueLen = len;
    return 0;
}

static HcclMemHandle g_userMemHandle = reinterpret_cast<HcclMemHandle>(0x1111);
static HcclMemHandle g_symMemHandle = reinterpret_cast<HcclMemHandle>(0x2222);
static ChannelHandle g_testChannel = static_cast<ChannelHandle>(0x3333);
static void *g_testDevChannelEntity = reinterpret_cast<void *>(0x5678);
// UT stub state, used in single-threaded test execution and reset in SetUp.
static HcclChannelDesc g_capturedChannelDesc {};
static std::vector<HcclMemHandle> g_capturedMemHandles;

HcclResult StubRegisterPendingSymmetricMemHandles(std::vector<HcclMemHandle> &memHandles)
{
    memHandles.clear();
    memHandles.emplace_back(g_symMemHandle);
    return HCCL_SUCCESS;
}

HcclResult StubRegisterPendingSymmetricMemHandlesEmpty(std::vector<HcclMemHandle> &memHandles)
{
    memHandles.clear();
    return HCCL_SUCCESS;
}

HcclResult StubCreateChannelsCapture(MyRank *self, CommEngine engine, const std::string &commTag,
    const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle *channels)
{
    (void)self;
    (void)engine;
    (void)commTag;
    if (channelNum > 0) {
        g_capturedChannelDesc = channelDescs[0];
        g_capturedMemHandles.clear();
        for (uint32_t idx = 0; idx < channelDescs[0].memHandleNum; ++idx) {
            g_capturedMemHandles.emplace_back(channelDescs[0].memHandles[idx]);
        }
        channels[0] = g_testChannel;
    }
    return HCCL_SUCCESS;
}

HcclResult StubChannelGet(ChannelHandle channelHandle, void **channel)
{
    EXPECT_EQ(channelHandle, g_testChannel);
    *channel = reinterpret_cast<void *>(g_testChannel);
    return HCCL_SUCCESS;
}

HcclResult StubBuildChannelEntityToDevice(AivUrmaChannel *self, void **devChannelEntity)
{
    EXPECT_EQ(reinterpret_cast<ChannelHandle>(self), g_testChannel);
    *devChannelEntity = g_testDevChannelEntity;
    return HCCL_SUCCESS;
}

HcclResult StubChannelGetRemoteMems(ChannelHandle channel, uint32_t *memNum, CommMem **remoteMems, char ***memTags)
{
    static CommMem remoteMem {};
    static char memTag[] = "__hccl_sym_win__ut";
    static char *tagList[] = {memTag};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x4444);
    remoteMem.size = 0x1000;
    EXPECT_EQ(channel, g_testChannel);
    *memNum = 1;
    *remoteMems = &remoteMem;
    *memTags = tagList;
    return HCCL_SUCCESS;
}

HcclResult StubUpdateSymmetricRemoteMem(uint32_t remoteRank, const CommMem *remoteMems, char **memTags, uint32_t memNum)
{
    (void)remoteRank;
    (void)remoteMems;
    (void)memTags;
    (void)memNum;
    return HCCL_SUCCESS;
}

class HcclChannelDescTest : public testing::Test {
public:
    void SetUp() override
    {
        g_capturedChannelDesc = {};
        g_capturedMemHandles.clear();
        g_testChannel = static_cast<ChannelHandle>(0x3333);
        const char *fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
        MOCKER(&HcclCommDfx::ReportKernel).stubs().will(returnValue(HCCL_SUCCESS));
        SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    void TearDown() override
    {
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

    void InitRoceChannelDesc(HcclChannelDesc &channelDesc) const
    {
        ASSERT_EQ(HcclChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
        channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
        channelDesc.roceAttr.queueNum = 3;
        channelDesc.roceAttr.retryCnt = 3;
        channelDesc.roceAttr.retryInterval = 20;
        channelDesc.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        channelDesc.localEndpoint.loc.device.devPhyId = 0U;
    }

    void ExpectRoceSlTcInHcommChannelDesc(uint32_t hcclQos, uint8_t expectSl, uint8_t expectTc)
    {
        hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(hcclQos);

        HcclChannelDesc in{};
        HcclChannelDesc out{};
        InitRoceChannelDesc(in);
        ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);

        ret = ProcessHcclResPackReq(in, out, hcclCommPtr.get());
        ASSERT_EQ(ret, HCCL_SUCCESS);

        const HcommChannelDesc hcommDesc = MyRankUtils::ChannelDescHccl2Hcomm(
            out, hcclCommPtr->GetCollComm()->GetCommConfig());
        EXPECT_EQ(hcommDesc.roceAttr.sl, expectSl) << "hcclQos=" << hcclQos;
        EXPECT_EQ(hcommDesc.roceAttr.tc, expectTc) << "hcclQos=" << hcclQos;
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
    
    // Mock BuildConnection 失败
    MOCKER(&HostCpuRoceChannel::BuildConnection).stubs().will(returnValue(HCCL_E_NETWORK));
    
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_IbvPostRecv_Fails_Return_Error)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    
    // Mock BuildConnection 成功
    MOCKER(&HostCpuRoceChannel::BuildConnection).stubs().will(returnValue(HCCL_SUCCESS));
    // Mock IbvPostRecv 失败
    MOCKER(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_E_INTERNAL));
    
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDesc.data(), 1, channels.data());
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

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_AicpuUrma_AppendSymmetricMemHandle)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    ASSERT_EQ(HcclChannelDescInit(channelDesc.data(), 1), HCCL_SUCCESS);
    channelDesc[0].remoteRank = 2;
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UBOE;
    channelDesc[0].notifyNum = 1;
    channelDesc[0].memHandles = &g_userMemHandle;
    channelDesc[0].memHandleNum = 1;

    MOCKER(&hcomm::ClusterMonitor::RegisterToClusterMonitor).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::RegisterPendingSymmetricMemHandles).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::CreateChannels).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_CpuUrma_NotAppendSymmetricMemHandle)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    ASSERT_EQ(HcclChannelDescInit(channelDesc.data(), 1), HCCL_SUCCESS);
    channelDesc[0].remoteRank = 2;
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UBOE;
    channelDesc[0].notifyNum = 1;
    channelDesc[0].memHandles = &g_userMemHandle;
    channelDesc[0].memHandleNum = 1;

    MOCKER_CPP(&CollComm::RegisterPendingSymmetricMemHandles).expects(never());
    MOCKER_CPP(&MyRank::CreateChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommDpuChannelRegisterDfx).stubs().will(returnValue(0));

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_AivUbg_ConvertsHostHandleToDeviceEntity)
{
    HcclChannelDesc channelDesc{};
    ASSERT_EQ(HcclChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteRank = 2;
    channelDesc.channelProtocol = COMM_PROTOCOL_UBG;

    hcomm::AivUrmaChannel channel(reinterpret_cast<EndpointHandle>(0x1), HcommChannelDesc{});
    g_testChannel = reinterpret_cast<ChannelHandle>(&channel);

    MOCKER(&hcomm::ClusterMonitor::RegisterToClusterMonitor).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::CreateChannels).stubs().will(invoke(StubCreateChannelsCapture));
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGet)
        .expects(once())
        .will(invoke(StubChannelGet));
    MOCKER_CPP(&hcomm::AivUrmaChannel::BuildChannelEntityToDevice,
        HcclResult(hcomm::AivUrmaChannel::*)(void **))
        .expects(once())
        .will(invoke(StubBuildChannelEntityToDevice));
    MOCKER_CPP(&hcomm::ChannelProcess::RegisterChannelD2HMap)
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));

    ChannelHandle outputChannel = 0;
    ret = HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AIV, &channelDesc, 1, &outputChannel);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(outputChannel, static_cast<ChannelHandle>(reinterpret_cast<uintptr_t>(g_testDevChannelEntity)));
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_HcclQosUnset_UsesRdmaEnvSlTc)
{
    // EnvRdmaConfig 默认 SL=4、TC=132（CI 中 setenv 晚于 EnvConfig 解析时不生效）
    ExpectRoceSlTcInHcommChannelDesc(HCCL_COMM_QOS_CONFIG_NOT_SET, 4U, 132U);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_HcclQos1_MapsSlAndTcFromHccn)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgRoceQosDscp));
    ExpectRoceSlTcInHcommChannelDesc(1U, 1U, static_cast<uint8_t>(20U << 2U));
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_HcclQos4_MapsSlAndTcFromHccn)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgRoceQosDscp));
    ExpectRoceSlTcInHcommChannelDesc(4U, 4U, static_cast<uint8_t>(50U << 2U));
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_HcclQos6_MapsSlAndTcFromHccn)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgRoceQosDscp));
    ExpectRoceSlTcInHcommChannelDesc(6U, 6U, static_cast<uint8_t>(70U << 2U));
}
