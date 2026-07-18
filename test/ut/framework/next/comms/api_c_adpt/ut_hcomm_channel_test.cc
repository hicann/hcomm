/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"

class TestHcommChannel : public TestHcommCAdptBase {
public:
    void SetUp() override {
        TestHcommCAdptBase::SetUp();
    }
    void TearDown() override {
        TestHcommCAdptBase::TearDown();
    }
};

TEST_F(TestHcommChannel, Ut_TestHcommChannelCreate_When_DescsNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle channels[1];
    HcommResult ret = HcommChannelCreate(nullptr, COMM_ENGINE_AICPU, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelCreate_When_ChannelsNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc desc;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 12345;
    HcommResult ret = HcommChannelCreate(nullptr, COMM_ENGINE_AICPU, &desc, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelCreate_When_NumZero_Return_HCCL_E_PARA)
{
    HcommChannelDesc desc;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 12345;
    ChannelHandle channels[1];
    HcommResult ret = HcommChannelCreate(nullptr, COMM_ENGINE_AICPU, &desc, 0, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommCollectiveChannelCreate_When_ParamsNull_Return_HCCL_E_PTR)
{
    ChannelHandle channels[1];
    HcommResult ret = HcommCollectiveChannelCreate(nullptr, COMM_ENGINE_AICPU, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommCollectiveChannelCreate_When_NumZero_Return_HCCL_E_PARA)
{
    HcommChannelDesc desc;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 12345;
    ChannelHandle channels[1];
    HcommResult ret = HcommCollectiveChannelCreate(nullptr, COMM_ENGINE_AICPU, &desc, 0, channels);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGet_When_ChannelNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelGet(0, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetStatus_When_ListNullptr_Return_HCCL_E_PTR)
{
    int32_t statusList[1] = { -1 };
    HcommResult ret = HcommChannelGetStatus(nullptr, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetStatus_When_StatusListNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle channelList[1] = { 0 };
    HcommResult ret = HcommChannelGetStatus(channelList, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetStatus_When_NumZero_Return_HCCL_E_PARA)
{
    ChannelHandle channelList[1] = { 0 };
    int32_t statusList[1] = { -1 };
    HcommResult ret = HcommChannelGetStatus(channelList, 0, statusList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetNotifyNum_When_NumNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelGetNotifyNum(0, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelDestroy_When_ChannelsNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelDestroy(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelDestroy_When_NumZero_Return_HCCL_E_PARA)
{
    ChannelHandle channels[1] = { 0 };
    HcommResult ret = HcommChannelDestroy(channels, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetRemoteMems_When_MemNullptr_Return_HCCL_E_PTR)
{
    uint32_t memNum = 0;
    char** memInfos = nullptr;
    HcommResult ret = HcommChannelGetRemoteMems(0, &memNum, nullptr, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetRemoteMems_When_NumNullptr_Return_HCCL_E_PTR)
{
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    HcommResult ret = HcommChannelGetRemoteMems(0, nullptr, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
