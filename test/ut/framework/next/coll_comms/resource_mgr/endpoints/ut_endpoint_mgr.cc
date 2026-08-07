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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "endpoint_mgr.h"
#include "hcomm_res_defs.h"

using namespace hcomm;

class EndpointMgrTest : public testing::Test {
protected:
    EndpointMgr mgr;

    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(EndpointMgrTest, IsMemExist_Empty) { EXPECT_FALSE(mgr.IsMemExist((EndpointHandle)0x1)); }

TEST_F(EndpointMgrTest, IsDescExist_Empty)
{
    EndpointDesc desc{};
    memset(&desc, 0, sizeof(desc));
    EXPECT_FALSE(mgr.IsDescExist(desc));
}

TEST_F(EndpointMgrTest, GetAllRegisteredMemory_NotExist)
{
    std::vector<MemHandle> memHandles;
    HcclResult ret = mgr.GetAllRegisteredMemory((EndpointHandle)0x1, memHandles);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(EndpointMgrTest, AddMemHandle_EmptyVec)
{
    std::vector<MemHandle> emptyVec;
    HcclResult ret = mgr.AddMemHandle((EndpointHandle)0x1, emptyVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(EndpointMgrTest, AddMemHandle_NewEndpoint)
{
    std::vector<MemHandle> handles = {(MemHandle)0x10, (MemHandle)0x20};
    HcclResult ret = mgr.AddMemHandle((EndpointHandle)0x1, handles);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(mgr.IsMemExist((EndpointHandle)0x1));
}

TEST_F(EndpointMgrTest, AddMemHandle_ExistingEndpoint)
{
    std::vector<MemHandle> handles1 = {(MemHandle)0x10};
    std::vector<MemHandle> handles2 = {(MemHandle)0x20, (MemHandle)0x30};

    HcclResult ret = mgr.AddMemHandle((EndpointHandle)0x1, handles1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = mgr.AddMemHandle((EndpointHandle)0x1, handles2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(mgr.IsMemExist((EndpointHandle)0x1));

    std::vector<MemHandle> allMems;
    ret = mgr.GetAllRegisteredMemory((EndpointHandle)0x1, allMems);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(allMems.size(), 3u);
}

TEST_F(EndpointMgrTest, GetAllRegisteredMemory_Success)
{
    std::vector<MemHandle> handles = {(MemHandle)0x10, (MemHandle)0x20, (MemHandle)0x30};
    mgr.AddMemHandle((EndpointHandle)0x1, handles);

    std::vector<MemHandle> result;
    HcclResult ret = mgr.GetAllRegisteredMemory((EndpointHandle)0x1, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], (MemHandle)0x10);
    EXPECT_EQ(result[1], (MemHandle)0x20);
    EXPECT_EQ(result[2], (MemHandle)0x30);
}

TEST_F(EndpointMgrTest, Get_CreateNewEndpoint)
{
    EndpointDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.protocol = COMM_PROTOCOL_HCCS;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr.s_addr = inet_addr("10.0.0.1");
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    EndpointHandle handle = nullptr;
    HcclResult ret = mgr.Get(desc, handle);
    EXPECT_NE(ret, HCCL_SUCCESS);
}
