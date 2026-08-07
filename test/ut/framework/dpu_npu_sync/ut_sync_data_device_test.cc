/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../communicator/hccl_api_base_test.h"
#include "aicpu_ts_sync_data_c_adpt.h"

#include <chrono>
#include <thread>
#include "thread/thread_aicpu_ts.h"
#include "thread/thread.h"

using namespace hccl;

constexpr size_t SHMEM_SIZE_BYTE = 512;
constexpr size_t MSG_TAG_SIZE_BYTE = 256;

class SyncDataDeviceTest : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

#pragma pack(push)
#pragma pack(1)
    struct MsgHeader {
        uint8_t flag;
        char msgTag[MSG_TAG_SIZE_BYTE];
        uint32_t msgId;
    };
#pragma pack(pop)
};

TEST_F(SyncDataDeviceTest, ut_HcommSendRequest_When_Normal_Expect_ReturnIsHCCL_SUCCESS_And_MemoryIsCorrect)
{
    void* devShmem = malloc(SHMEM_SIZE_BYTE);

    MsgHandle handle = reinterpret_cast<MsgHandle>(devShmem);
    const char msgTag[MSG_TAG_SIZE_BYTE] = "Hello HCCL";
    const char data[] = "Open Source is Good.";
    const size_t dataSizeByte = sizeof(data);
    uint32_t outMsgId = 0;
    int32_t ret = HCCL_E_RESERVED;

    ret = HcommSendRequest(handle, msgTag, data, dataSizeByte, &outMsgId);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    MsgHeader* structedDevShmem = static_cast<MsgHeader*>(devShmem);

    printf(
        "Simulated Device Shared Mem: [ %u | %s | %u | %s ]\n", structedDevShmem->flag, structedDevShmem->msgTag,
        structedDevShmem->msgId, static_cast<char*>(devShmem) + sizeof(MsgHeader));

    EXPECT_EQ(structedDevShmem->flag, 1);
    EXPECT_STREQ(structedDevShmem->msgTag, msgTag);
    EXPECT_EQ(structedDevShmem->msgId, outMsgId);
    EXPECT_STREQ(static_cast<char*>(devShmem) + sizeof(MsgHeader), data);

    free(devShmem);
    devShmem = nullptr;
}

TEST_F(SyncDataDeviceTest, ut_HcommSendRequest_When_HandleIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    void* devShmem = nullptr;

    MsgHandle handle = reinterpret_cast<MsgHandle>(devShmem);
    const char msgTag[MSG_TAG_SIZE_BYTE] = "Hello HCCL";
    const char data[] = "Open Source is Good.";
    const size_t dataSizeByte = sizeof(data);
    uint32_t outMsgId = 0;
    int32_t ret = HCCL_E_RESERVED;

    ret = HcommSendRequest(handle, msgTag, data, dataSizeByte, &outMsgId);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(SyncDataDeviceTest, ut_HcommWaitResponse_When_Normal_Expect_ReturnIsHCCL_SUCCESS_And_ResultIsCorrect)
{
    void* devShmem = malloc(SHMEM_SIZE_BYTE);
    memset_s(devShmem, SHMEM_SIZE_BYTE, 0, SHMEM_SIZE_BYTE);

    const char dpuData[] = "Open Source is Good.";
    const size_t dpuDataSizeByte = sizeof(dpuData);
    const uint32_t dpuMsgId = 1145;

    std::thread dpuKernel([=]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        MsgHeader* structedDevShmem = reinterpret_cast<MsgHeader*>(devShmem);

        strcpy_s(reinterpret_cast<char*>(structedDevShmem) + sizeof(MsgHeader), dpuDataSizeByte, dpuData);
        strcpy_s(structedDevShmem->msgTag, MSG_TAG_SIZE_BYTE, "DPU Msg");
        structedDevShmem->msgId = dpuMsgId;
        structedDevShmem->flag = 1;

        printf("Dpu Kernel End.\n");
    });

    MsgHandle handle = reinterpret_cast<MsgHandle>(devShmem);
    char dst[SHMEM_SIZE_BYTE] = "";
    uint32_t outMsgId = 0;
    int32_t ret = HCCL_E_RESERVED;

    ret = HcommWaitResponse(handle, dst, dpuDataSizeByte, &outMsgId);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    printf("dst: %s\n", dst);

    EXPECT_STREQ(dst, dpuData);
    EXPECT_EQ(outMsgId, dpuMsgId);
    EXPECT_EQ(static_cast<MsgHeader*>(devShmem)->flag, 0); // flag is resetted to 0

    dpuKernel.join();

    free(devShmem);
    devShmem = nullptr;
}

TEST_F(SyncDataDeviceTest, ut_HcommThreadSynchronize_When_ThreadIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    ThreadHandle thread = nullptr;
    int32_t ret = HcommThreadSynchronize(thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(SyncDataDeviceTest, ut_HcommThreadSynchronize_When_ThreadIsValid_Expect_ReturnIsHCCL_SUCCESS)
{
    hccl::Thread* threadPtr = new (std::nothrow) hccl::AicpuTsThread(nullptr);
    ASSERT_NE(threadPtr, nullptr);

    ThreadHandle thread = reinterpret_cast<ThreadHandle>(threadPtr);
    int32_t ret = HcommThreadSynchronize(thread);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete threadPtr;
    threadPtr = nullptr;
}
