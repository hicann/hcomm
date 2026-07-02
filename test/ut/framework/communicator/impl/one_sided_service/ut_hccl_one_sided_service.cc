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
#include <mockcpp/mockcpp.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <securec.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>

#include <sys/types.h>
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>

#define private public
#define protected public
#include "adapter_rts.h"
#include "exception_handler.h"
#include "hccl_socket_manager.h"
#include "notify_pool.h"
#include "i_hccl_one_sided_service.h"
#include "hccl_one_sided_service.h"
#include <hccl/hccl_one_sided_services.h>
#include "hccl_one_sided_conn.h"
#include "rma_buffer_mgr.h"
#include "local_rdma_rma_buffer.h"
#include "local_ipc_rma_buffer.h"
#include "remote_rdma_rma_buffer.h"
#include "remote_ipc_rma_buffer.h"
#include "local_rdma_rma_buffer_impl.h"
#include "local_ipc_rma_buffer_impl.h"
#include "remote_rdma_rma_buffer_impl.h"
#include "remote_ipc_rma_buffer_impl.h"
#include "hccl_network.h"
#include "transport_roce_mem.h"
#include "transport_ipc_mem.h"
#include "dispatcher_pub.h"
#include "op_base.h"
#include "global_mem_record.h"
#include "global_mem_manager.h"
#include "externalinput.h"
#undef protected
#undef private

using namespace std;
using namespace hccl;

class OneSidedUt : public testing::Test
{
protected:
    void SetUp() override {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum)
            .stubs()
            .with(mockcpp::any(), outBound(portNum))
            .will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }

    void TearDown() override {
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(OneSidedUt, Prepare_When_PrepareFullMeshFail_With_NullptrConn_Expect_SkipClean)
{
    unique_ptr<HcclSocketManager> socketManager = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::PrepareFullMesh)
    .stubs()
    .will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&HcclOneSidedConn::CleanSocketResource)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    // 构造ranktable
    HcclDispatcher dispatcher = &notifyPool;
    HcclRankLinkInfo localRankInfo{};
    localRankInfo.userRank = 0;
    RankTable_t rankTable{};
    RankInfo_t rankInfo0;
    rankInfo0.rankId = 0;
    RankInfo_t rankInfo1;
    rankInfo1.rankId = 1;
    rankTable.rankList.push_back(rankInfo0);
    rankTable.rankList.push_back(rankInfo1);
    service->Config(dispatcher, localRankInfo, &rankTable);

    service->oneSidedConns_[1] = nullptr;

    std::string commIdentifier("test");
    HcclPrepareConfig config;
    config.topoType = HcclTopoType::HCCL_TOPO_FULLMESH;
    HcclResult ret = service->Prepare(commIdentifier, &config, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, HcclRegisterGlobalMem_ValidMem_Expect_ReinterpretCastCovered)
{
    auto buffer = std::vector<int8_t>(10);
    CommMem mem{COMM_MEM_TYPE_DEVICE, buffer.data(), buffer.size()};
    void* memHandle = nullptr;
    HcclResult ret = HcclRegisterGlobalMem(&mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(memHandle, nullptr);

    HcclDeregisterGlobalMem(memHandle);
}

TEST_F(OneSidedUt, HcclRemapRegistedMemory_ValidParams_Expect_ReinterpretCastCovered)
{
    auto commObj = std::make_unique<hccl::hcclComm>();
    void* commArr[1] = {static_cast<void*>(commObj.get())};

    unique_ptr<HcclSocketManager> socketManager = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    hccl::HcclOneSidedService service(socketManager, notifyPool, commConfig);

    hccl::IHcclOneSidedService* servicePtr = &service;

    CommMem memInfoArray[1];
    memInfoArray[0].type = COMM_MEM_TYPE_DEVICE;
    memInfoArray[0].addr = reinterpret_cast<void*>(0x1000);
    memInfoArray[0].size = 1024;

    MOCKER_CPP(&hccl::hcclComm::GetOneSidedService)
    .stubs()
    .with(outBoundP(&servicePtr))
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclRemapRegistedMemory(commArr, memInfoArray, 1, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commObj.reset();
    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, HcclEnableMemAccess_LegacyPath_Expect_ReinterpretCastCovered)
{
    auto commObj = std::make_unique<hccl::hcclComm>();

    unique_ptr<HcclSocketManager> socketManager = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    hccl::HcclOneSidedService service(socketManager, notifyPool, commConfig);

    hccl::IHcclOneSidedService* servicePtr = &service;

    MOCKER_CPP(&hccl::hcclComm::GetOneSidedService)
    .stubs()
    .with(outBoundP(&servicePtr))
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::EnableMemAccess, void(HcclOneSidedService::*)(const HcclMemDesc &, HcclMem &))
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclMemDesc remoteMemDesc{};
    CommMem remoteMem{};
    HcclResult ret = HcclEnableMemAccess(static_cast<HcclComm>(commObj.get()), &remoteMemDesc, &remoteMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commObj.reset();
    GlobalMockObject::verify();
}