/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FIT FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include <memory>
#include <string>
#include <cstring>
#include "hccl_types.h"
#define private public
#include "dfx/cluster_monitor/cluster_monitor.h"
#undef private
#include "hccl_comm_socket_c_adpt.h"
#include "base_config.h"
#include "llt_hccl_stub_rank_graph.h"
#include "hccl/hccl_comm.h"

using namespace hcomm;

class ClusterMonitorTest : public ::testing::Test {
public:
    void SetUp() override {
        testing::Test::SetUp();
    }
    void TearDown() override {
        testing::Test::TearDown();
        GlobalMockObject::verify();
    }

    static ClusterMonitor g_monitor;
};

ClusterMonitor ClusterMonitorTest::g_monitor;

TEST_F(ClusterMonitorTest, Ut_CreateMonitorLinksAsync_When_NormalInput_Expect_CreateLinks)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    s32 ret;
    SocketDesc socketDesc;
    std::string testTag = "monitor_test_tag";
    struct in_addr localAddr;
    struct in_addr remoteAddr;
    inet_pton(AF_INET, "127.0.0.1", &localAddr);
    socketDesc.localEndpoint.commAddr.addr = localAddr;
    socketDesc.localEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc.localEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    socketDesc.localEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    inet_pton(AF_INET, "192.186.0.1", &remoteAddr);
    socketDesc.remoteEndpoint.commAddr.addr = remoteAddr;
    socketDesc.remoteEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc.remoteEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    socketDesc.remoteEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    ret = memcpy_s(socketDesc.tag, HCCL_SOCKET_TAG_LEN, testTag.c_str(), testTag.length() + 1);
    EXPECT_EQ(ret, EOK);
    socketDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
    socketDesc.listenPort = 8080;

    SocketDesc socketDesc1;
    testTag = "monitor_test_tag_1";
    inet_pton(AF_INET, "127.0.0.1", &localAddr);
    socketDesc1.localEndpoint.commAddr.addr = localAddr;
    socketDesc1.localEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc1.localEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_HOST;
    socketDesc1.localEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    inet_pton(AF_INET, "192.186.0.2", &remoteAddr);
    socketDesc1.remoteEndpoint.commAddr.addr = remoteAddr;
    socketDesc1.remoteEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc1.remoteEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    socketDesc1.remoteEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    ret = memcpy_s(socketDesc1.tag, HCCL_SOCKET_TAG_LEN, testTag.c_str(), testTag.length() + 1);
    EXPECT_EQ(ret, EOK);
    socketDesc1.role = HCOMM_SOCKET_ROLE_CLIENT;
    socketDesc1.listenPort = 8080;

    SocketDesc socketDesc2;
    testTag = "monitor_test_tag_2";
    inet_pton(AF_INET, "127.0.0.1", &localAddr);
    socketDesc2.localEndpoint.commAddr.addr = localAddr;
    socketDesc2.localEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc2.localEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_HOST;
    socketDesc2.localEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    inet_pton(AF_INET, "192.186.0.3", &remoteAddr);
    socketDesc2.remoteEndpoint.commAddr.addr = remoteAddr;
    socketDesc2.remoteEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc2.remoteEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_HOST;
    socketDesc2.remoteEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    ret = memcpy_s(socketDesc2.tag, HCCL_SOCKET_TAG_LEN, testTag.c_str(), testTag.length() + 1);
    EXPECT_EQ(ret, EOK);
    socketDesc2.role = HCOMM_SOCKET_ROLE_CLIENT;
    socketDesc2.listenPort = 8080;

    ClusterMonitorSocketCtx context;
    context.socketDesc = socketDesc;
    context.socketHandler = nullptr;

    ClusterMonitorSocketCtx context1;
    context1.socketDesc = socketDesc1;
    context1.socketHandler = nullptr;

    ClusterMonitorSocketCtx context2;
    context2.socketDesc = socketDesc2;
    context2.socketHandler = nullptr;

    ClusterUIDType uid;
    ret = memcpy_s(uid.id, sizeof(uid.id), "test_uid", sizeof("test_uid") + 1);
    EXPECT_EQ(ret, EOK);

    ClusterUIDType uid1;
    ret = memcpy_s(uid1.id, sizeof(uid1.id), "test_uid1", sizeof("test_uid1") + 1);
    EXPECT_EQ(ret, EOK);

    ClusterUIDType uid2;
    ret = memcpy_s(uid2.id, sizeof(uid2.id), "test_uid2", sizeof("test_uid2") + 1);
    EXPECT_EQ(ret, EOK);
    
    g_monitor.commIdMap_["comm1"].insert(std::make_pair(uid, false));
    g_monitor.commIdMap_["comm1"].insert(std::make_pair(uid1, false));
    g_monitor.commIdMap_["comm1"].insert(std::make_pair(uid2, false));

    std::unique_lock<std::mutex> linksLock(g_monitor.clusertMonitorLinkMtx_);
    g_monitor.clusterLinkContext_["comm1"].push(std::make_pair(uid, context));
    g_monitor.clusterLinkContext_["comm1"].push(std::make_pair(uid1, context1));
    g_monitor.clusterLinkContext_["comm2"].push(std::make_pair(uid2, context2));
    linksLock.unlock();

    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().with(any()).will(returnValue((s32)(5)));
    g_monitor.CreateHBLinksAsync();

    u32 uncompletedCount = 3;
    auto linkThreadIt = g_monitor.monitorLinkStatusMap_.begin();
    const auto CREATE_LINK_TIMEOUT = std::chrono::seconds(10);
    auto startTime = std::chrono::steady_clock::now();
    while((std::chrono::steady_clock::now() - startTime) <= CREATE_LINK_TIMEOUT) {
        HCCL_ERROR("Waiting for monitor links to complete, uncompletedCount = %u", uncompletedCount);
        if (linkThreadIt == g_monitor.monitorLinkStatusMap_.end()) {
            linkThreadIt = g_monitor.monitorLinkStatusMap_.begin();
        } 
        if (linkThreadIt->second == ClusterMonitor::MonitorLinkStatus::MONITOR_LINK_COMPLETED) {
            uncompletedCount--;
        }
        if (uncompletedCount == 0) {
            break;
        } else {
            linkThreadIt++;
        }
    }

    g_monitor.linkThreadRunning_ = false;
    // 在心跳进程结束之前join所有的建链线程
    for (auto &pair : g_monitor.linkThreadMap_) {
        if (pair.second != nullptr && pair.second->joinable()) {
            pair.second->join();
            HCCL_INFO("[HeartbeatStatusMonitor] thread has joined. Remote uid is [%s]", g_monitor.GetUID(pair.first).c_str());
        }
    }

    EXPECT_EQ(uncompletedCount, 0);
}

TEST_F(ClusterMonitorTest, Ut_SendMonitorFrame_When_NormalInput_Expect_SendFrame)
{
    for (auto iter = g_monitor.uid2SocketRefMap_.begin(); iter != g_monitor.uid2SocketRefMap_.end(); iter++) {
        ClusterUIDType rem = iter->first;
        g_monitor.uid2SocketRefMap_[rem].lostNum++;
        g_monitor.SendFrame(rem, g_monitor.myRankUID_, g_monitor.myRankUID_, ClusterMonitorStatus::CLUSTER_MONITOR_STUCK);
        EXPECT_EQ(g_monitor.uid2SocketRefMap_[rem].sendBuffer.size(), 0);
    }
}

TEST_F(ClusterMonitorTest, Ut_RecvMonitorFrame_When_NormalInput_Expect_RecvFrame)
{
    MOCKER_CPP(&ClusterMonitor::ParseFrame)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    uint64_t compSize = 0;
    MOCKER(SocketRecvNb)
    .stubs()
    .with(any(), any(), any(), outBoundP(&compSize, sizeof(uint64_t *)))
    .will(invoke(SocketRecvNb))    // 第1次走真实函数
    .then(returnValue(HCCL_E_AGAIN));  // 第2次返回特定值

    for (auto iter = g_monitor.uid2SocketRefMap_.begin(); iter != g_monitor.uid2SocketRefMap_.end(); iter++) {
        ClusterUIDType rem = iter->first;
        HcclResult ret = g_monitor.RecvFrame(rem);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}

TEST_F(ClusterMonitorTest, Ut_GetCqeErrInfoFromTaskException_When_NormalInput_Expect_SetCqeErrInfo)
{
    std::string netInstId = "localInstance";
    u32 localId = 0;
    ClusterUIDCxt localUidCxt(netInstId, localId);
    ClusterUIDType localUid = g_monitor.FormatUID(localUidCxt);
    g_monitor.myRankUID_ = localUid;
    g_monitor.myRankNetInstId_ = netInstId;
    g_monitor.myRankLocalId_ = localId;

    u32 remoteLocalId = 1;
    uint16_t status = 1;
    std::string localEid = "localEid";
    std::string remoteEid = "remoteEid";
    std::string remoteInsId = "remoteInsId";

    g_monitor.GetCqeErrInfoFromTaskException(remoteLocalId, status, localEid, remoteEid, remoteInsId);

    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeRemoteLocalId, remoteLocalId);
    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeStatus, status);
    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeLocalEid, localEid);
    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeRemoteEid, remoteEid);
    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeRemoteInsId, remoteInsId);
}

TEST_F(ClusterMonitorTest, Ut_ProcessExceptionEvent_When_NormalQueue_Expect_SendFrames)
{
    std::string netInstId = "localInstance";
    u32 localId = 0;
    ClusterUIDCxt localUidCxt(netInstId, localId);
    ClusterUIDType localUid = g_monitor.FormatUID(localUidCxt);
    g_monitor.myRankUID_ = localUid;

    std::string crimerInstId = "remoteInstance";
    ClusterUIDCxt crimerCxt(crimerInstId, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    ClusterUIDType informerUid = localUid;

    ClusterMonitor::FrameStatus frameStatus;
    frameStatus.status = ClusterMonitorStatus::CLUSTER_MONITOR_LOST;
    frameStatus.informer = informerUid;
    frameStatus.needBroadcast = false;
    g_monitor.uid2FrameStatusMap_.insert(crimerUid, frameStatus);

    g_monitor.errRankQueue_.push(crimerUid);

    std::string otherInstId = "otherInstance";
    ClusterUIDCxt otherCxt(otherInstId, 2);
    ClusterUIDType otherUid = g_monitor.FormatUID(otherCxt);

    ClusterMonitorSocketCtx socketCtx;
    socketCtx.socketHandler = nullptr;
    g_monitor.uid2SocketRefMap_.insert(otherUid, socketCtx);

    ClusterMonitor::FrameStatus otherStatus;
    otherStatus.status = ClusterMonitorStatus::CLUSTER_MONITOR_OK;
    g_monitor.uid2FrameStatusMap_.insert(otherUid, otherStatus);

    MOCKER(SocketSendNb)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    g_monitor.ProcessExceptionEvent();

    EXPECT_TRUE(g_monitor.errRankQueue_.empty());

    GlobalMockObject::verify();
}

TEST_F(ClusterMonitorTest, Ut_ProcessExceptionEvent_When_EmptyQueue_Expect_Return)
{
    while (!g_monitor.errRankQueue_.empty()) {
        g_monitor.errRankQueue_.pop();
    }

    g_monitor.ProcessExceptionEvent();

    EXPECT_TRUE(g_monitor.errRankQueue_.empty());
}

TEST_F(ClusterMonitorTest, Ut_MakeErrMsg_When_LostStatus_Expect_GenerateCorrectMsg)
{
    std::queue<ClusterMonitorFrame> keyEvents;

    std::string crimerStr = "crimerInstance";
    ClusterUIDCxt crimerCxt(crimerStr, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    std::string informerStr = "informerInstance";
    ClusterUIDCxt informerCxt(informerStr, 2);
    ClusterUIDType informerUid = g_monitor.FormatUID(informerCxt);

    HcclUs relativeTime = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    ClusterMonitorFrame frame(crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_LOST, relativeTime, systemTime);
    keyEvents.push(frame);

    std::vector<std::string> errStatusVec;
    g_monitor.MakeErrMsg(keyEvents, errStatusVec);

    EXPECT_EQ(errStatusVec.size(), 1);
    EXPECT_TRUE(errStatusVec[0].find("Heartbeat Lost Occurred") != std::string::npos);
}

TEST_F(ClusterMonitorTest, Ut_MakeErrMsg_When_CqeErrStatus_Expect_GenerateCorrectMsg)
{
    std::queue<ClusterMonitorFrame> keyEvents;

    std::string crimerStr2 = "crimerInstance";
    ClusterUIDCxt crimerCxt(crimerStr2, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    std::string informerStr2 = "informerInstance";
    ClusterUIDCxt informerCxt(informerStr2, 2);
    ClusterUIDType informerUid = g_monitor.FormatUID(informerCxt);

    HcclUs relativeTime = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    ClusterMonitorFrame frame(crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR, relativeTime, systemTime);
    keyEvents.push(frame);

    std::vector<std::string> errStatusVec;
    g_monitor.MakeErrMsg(keyEvents, errStatusVec);

    EXPECT_EQ(errStatusVec.size(), 1);
    EXPECT_TRUE(errStatusVec[0].find("Error cqe Occurred") != std::string::npos);
}

TEST_F(ClusterMonitorTest, Ut_MakeErrMsg_When_EmptyQueue_Expect_EmptyVec)
{
    std::queue<ClusterMonitorFrame> keyEvents;
    std::vector<std::string> errStatusVec;

    g_monitor.MakeErrMsg(keyEvents, errStatusVec);

    EXPECT_EQ(errStatusVec.size(), 0);
}

TEST_F(ClusterMonitorTest, Ut_PrintEvents_When_NormalMap_Expect_ReturnVec)
{
    std::map<ClusterMonitorStatus, std::queue<ClusterMonitorFrame>> keyEvents;

    std::string crimerStr3 = "crimerInstance";
    ClusterUIDCxt crimerCxt(crimerStr3, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    std::string informerStr3 = "informerInstance";
    ClusterUIDCxt informerCxt(informerStr3, 2);
    ClusterUIDType informerUid = g_monitor.FormatUID(informerCxt);

    HcclUs relativeTime = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    ClusterMonitorFrame cqeFrame(crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR, relativeTime, systemTime);
    keyEvents[ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR].push(cqeFrame);

    ClusterMonitorFrame lostFrame(crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_LOST, relativeTime, systemTime);
    keyEvents[ClusterMonitorStatus::CLUSTER_MONITOR_LOST].push(lostFrame);

    auto result = g_monitor.PrintEvents(keyEvents);

    EXPECT_GE(result.size(), 1);
}

TEST_F(ClusterMonitorTest, Ut_PrintEvents_When_EmptyMap_Expect_EmptyVec)
{
    std::map<ClusterMonitorStatus, std::queue<ClusterMonitorFrame>> keyEvents;

    auto result = g_monitor.PrintEvents(keyEvents);

    EXPECT_EQ(result.size(), 0);
}

TEST_F(ClusterMonitorTest, Ut_GetErrStatusVecFromCluserMonitor_When_Normal_Expect_ReturnVec)
{
    std::string crimerStr4 = "crimerInstance";
    ClusterUIDCxt crimerCxt(crimerStr4, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    std::string informerStr4 = "informerInstance";
    ClusterUIDCxt informerCxt(informerStr4, 2);
    ClusterUIDType informerUid = g_monitor.FormatUID(informerCxt);

    HcclUs relativeTime = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    ClusterMonitorFrame frame(crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR, relativeTime, systemTime);
    g_monitor.errStatusQueue_.push(frame);

    auto result = g_monitor.GetErrStatusVecFromCluserMonitor();

    EXPECT_EQ(g_monitor.errStatusQueue_.size(), 0);
}

TEST_F(ClusterMonitorTest, Ut_GetErrStatusVecFromCluserMonitor_When_EmptyQueue_Expect_EmptyVec)
{
    while (!g_monitor.errStatusQueue_.empty()) {
        g_monitor.errStatusQueue_.pop();
    }

    auto result = g_monitor.GetErrStatusVecFromCluserMonitor();

    EXPECT_EQ(result.size(), 0);
}


TEST_F(ClusterMonitorTest, Ut_MonitorThread_When_Normal_Expect_ProcessEvents)
{
    MOCKER_CPP(&ClusterMonitor::SendFrame)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&ClusterMonitor::RecvFrame)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&ClusterMonitor::PrintEvents)
        .stubs()
        .will(returnValue(std::vector<std::string>{"Error Message"}));

    HCCL_ERROR("g_monitor.uid2SocketRefMap_.Size = %d", g_monitor.uid2SocketRefMap_.Size());
    g_monitor.clusterMonitorThreadFlag_  = true;
    g_monitor.RunMonitorThread();
    EXPECT_TRUE(g_monitor.clusterMonitorThread_ != nullptr);
    SalSleep(1);  // 等待线程处理完队列中的事件
    g_monitor.clusterMonitorThreadFlag_  = false;

    if (g_monitor.clusterMonitorThread_) {
        if (g_monitor.clusterMonitorThread_->joinable()) {
            g_monitor.clusterMonitorThread_->join();
        }
    }
}

TEST_F(ClusterMonitorTest, Ut_RegisterToClusterMonitor_When_Normal_Expect_Success)
{
    Hccl::RankGraph stub(0);
    void *stubPtr = &stub;
    MOCKER_CPP(&Hccl::HcclCommunicator::GetRankGraphV2)
    .stubs()
    .with(outBound(stubPtr))
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&ClusterMonitor::GetRemEndpointDescs)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    bool isDeviceSide {
        false
    };
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_RDMA_RETRY_CNT", "7", 1);
    setenv("HCCL_RDMA_TIMEOUT", "20", 1);
    setenv("HCCL_RDMA_TC", "120", 1);
    setenv("HCCL_RDMA_SL", "2", 1);
    setenv("HCCL_DFS_CONFIG", "clusterHeartBeatEnable:on", 1);
    hccl::RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    void* commV2 = (void*)0x2000;
    uint32_t rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[128] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclOpExpansionMode = 1; // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level 
    unsetenv("HCCL_DFS_CONFIG");    
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    HcclComm comm = static_cast<HcclComm>(hcclCommPtr.get());

    ret = g_monitor.RegisterToClusterMonitor(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ClusterMonitorTest, Ut_UnRegisterToClusterMonitor_When_Normal_Expect_Success)
{
    MOCKER(SocketDestroy).stubs().will(returnValue(HCCL_SUCCESS));

    g_monitor.initialized_ = true;  // 设置已初始化标志，允许调用UnRegisterToClusterMonitor
    hccl::RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    void* commV2 = (void*)0x2000;
    uint32_t rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[128] = {"comm1"};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclOpExpansionMode = 1; // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level 
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    HcclComm comm = static_cast<HcclComm>(hcclCommPtr.get());

    std::string commKey(commName);
    std::string netInstId("remoteInstance");
    ClusterUIDCxt clusterUidCxt(netInstId, 1);
    ClusterUIDType clusterUid = g_monitor.FormatUID(clusterUidCxt);
    g_monitor.commIdMap_[commKey][clusterUid] = true;
    ret = g_monitor.UnRegisterToClusterMonitor(collComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}


