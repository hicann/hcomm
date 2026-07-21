/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * This SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../ut_hcomm_base.h"
#include "channel_process.h"
#include "aiv_urma_channel.h"
#include "aicpu_ts_roce_channel_v2.h"
#include "aicpu_ts_channel_helper.h"
#include "aiv_channel_helper.h"

class TestChannelProcess : public TestHcommCAdptBase {
public:
    void SetUp() override
    {
        TestHcommCAdptBase::SetUp();
    }
    void TearDown() override
    {
        {
            std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
            hcomm::ChannelProcess::g_ChannelMap.clear();
            hcomm::ChannelProcess::g_ChannelD2HMap.clear();
        }
        TestHcommCAdptBase::TearDown();
    }
};

// CreateChannelsLoop 空指针测试
TEST_F(TestChannelProcess, Ut_TestCreateChannelsLoop_When_EndpointHandleNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc desc = {};
    ChannelHandle outHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::CreateChannelsLoop(nullptr, COMM_ENGINE_AICPU, &desc, 1, outHandles);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// FillChannelD2HMap 空指针和参数校验测试
TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_DeviceChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(nullptr, hostHandles, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_HostChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle deviceHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(deviceHandles, nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_ListNumZero_Return_HCCL_E_PARA)
{
    ChannelHandle deviceHandles[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(deviceHandles, hostHandles, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// LaunchChannelKernelCommon 空指针和参数校验测试
TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_ChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        nullptr, hostHandles, hcommDescs, 1, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_HostChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        deviceHandles, nullptr, hcommDescs, 1, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_ListNumZero_Return_HCCL_E_PARA)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        deviceHandles, hostHandles, hcommDescs, 0, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ChannelGetRemoteMems 空指针测试
TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_RemoteMemNullptr_Return_HCCL_E_PTR)
{
    char **memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, &memNum, nullptr, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_MemInfosNullptr_Return_HCCL_E_PTR)
{
    CommMem *remoteMem = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, &memNum, &remoteMem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_MemNumNullptr_Return_HCCL_E_PTR)
{
    CommMem *remoteMem = nullptr;
    char **memInfos = nullptr;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, nullptr, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_ChannelClean_NullList_Returns_E_PARA)
{
    // Passing null pointer should return parameter error
    auto ret = hcomm::ChannelProcess::ChannelClean(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_ChannelResumeConcurrency_ZeroChannels_Returns_SUCCESS)
{
    // Zero channel count should be a no-op and return success
    ChannelHandle dummyList[1] = {0};
    auto ret = hcomm::ChannelProcess::ChannelResumeConcurrency(dummyList, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestChannelProcess, Ut_ChannelResume_When_ResumeConcurrencyFails_ReturnsError)
{
    ChannelHandle list[1] = {(ChannelHandle)0x1};
    // Mock ChannelResumeConcurrency to return internal error
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelResumeConcurrency, HcclResult(const ChannelHandle *, uint32_t))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    auto ret = hcomm::ChannelProcess::ChannelResume(list, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TestChannelProcess, Ut_ChannelResume_NullList_Returns_E_PARA)
{
    auto ret = hcomm::ChannelProcess::ChannelResume(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_LaunchChannelKernel_When_ChannelKindIsUBOE_CallsChannelKernelLaunchForBase)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};

    class FakeUboeChannel : public hcomm::Channel {
    public:
        hcomm::HcommChannelKind GetChannelKind() const override
        {
            return hcomm::HcommChannelKind::AICPU_TS_UBOE;
        }
        HcclResult Init() override
        {
            return HCCL_SUCCESS;
        }
        HcclResult GetNotifyNum(uint32_t *notifyNum) const override
        {
            return HCCL_SUCCESS;
        }
        HcclResult GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos) override
        {
            return HCCL_SUCCESS;
        }
        hcomm::ChannelStatus GetStatus() override
        {
            return hcomm::ChannelStatus::READY;
        }
        const HcommChannelDesc &GetChannelDesc() const override
        {
            static HcommChannelDesc defaultDesc{};
            return defaultDesc;
        }
        HcclResult Clean() override
        {
            return HCCL_SUCCESS;
        }
        HcclResult Resume() override
        {
            return HCCL_SUCCESS;
        }
        HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override
        {
            return HCCL_SUCCESS;
        }
        HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override
        {
            return HCCL_SUCCESS;
        }
        HcclResult WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx) override
        {
            return HCCL_SUCCESS;
        }
        HcclResult Write(void *dst, const void *src, uint64_t len) override
        {
            return HCCL_SUCCESS;
        }
        HcclResult Read(void *dst, const void *src, uint64_t len) override
        {
            return HCCL_SUCCESS;
        }
        HcclResult ChannelFence() override
        {
            return HCCL_SUCCESS;
        }
    };

    FakeUboeChannel fakeChannel;
    ChannelHandle hostHandles[1] = {reinterpret_cast<ChannelHandle>(&fakeChannel)};

    MOCKER_CPP(&hcomm::ChannelProcess::LaunchChannelKernelCommon,
        HcclResult(ChannelHandle *, ChannelHandle *, HcommChannelDesc *, uint32_t, const std::string &, aclrtBinHandle,
            const std::string &, bool))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto ret = hcomm::ChannelProcess::LaunchChannelKernel(deviceHandles, hostHandles, hcommDescs, 1, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ===================== 非阻塞建链 UT =====================

// 通用 FakeChannel，支持配置 engine/kind/status/deviceEntityReady/ctxMem
class FakeChannel : public hcomm::Channel {
public:
    FakeChannel(CommEngine eng = COMM_ENGINE_AICPU, hcomm::ChannelStatus sta = hcomm::ChannelStatus::READY)
    {
        engine_ = eng;
        channelKind_ = hcomm::HcommChannelKind::AICPU_TS_UBOE;
        fakeStatus_ = sta;
    }
    ~FakeChannel() = default;

    void SetStatus(hcomm::ChannelStatus s)
    {
        fakeStatus_ = s;
    }
    void SetDesc(const HcommChannelDesc &d)
    {
        desc_ = d;
    }

    hcomm::HcommChannelKind GetChannelKind() const override
    {
        return channelKind_;
    }
    HcclResult Init() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult GetNotifyNum(uint32_t *notifyNum) const override
    {
        return HCCL_SUCCESS;
    }
    HcclResult GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos) override
    {
        return HCCL_SUCCESS;
    }
    hcomm::ChannelStatus GetStatus() override
    {
        return fakeStatus_;
    }
    const HcommChannelDesc &GetChannelDesc() const override
    {
        return desc_;
    }
    HcclResult Clean() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Resume() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Write(void *dst, const void *src, uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Read(void *dst, const void *src, uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult ChannelFence() override
    {
        return HCCL_SUCCESS;
    }

private:
    hcomm::ChannelStatus fakeStatus_{hcomm::ChannelStatus::INIT};
    HcommChannelDesc desc_{};
};

// 辅助：注册 FakeChannel 到全局 map
static ChannelHandle RegisterFakeChannel(std::shared_ptr<FakeChannel> &ch)
{
    ChannelHandle handle = reinterpret_cast<ChannelHandle>(ch.get());
    int32_t deviceId = 0;
    // mock hrtGetDevice 返回 0
    hcomm::DeviceChannelKey key{0, handle};
    std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
    hcomm::ChannelProcess::g_ChannelMap.emplace(handle, ch);
    hcomm::ChannelProcess::g_ChannelD2HMap.emplace(key, handle);
    return handle;
}

// --- GetChannelsInfo 参数校验 ---
TEST_F(TestChannelProcess, Ut_GetChannelsInfo_When_Nullptr_Return_E_PTR)
{
    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> descs;
    std::vector<hcomm::ChannelStatus> status;
    EXPECT_EQ(hcomm::ChannelProcess::GetChannelsInfo(nullptr, 1, engines, descs, status), HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_GetChannelsInfo_When_ZeroNum_Return_E_PARA)
{
    ChannelHandle list[1] = {};
    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> descs;
    std::vector<hcomm::ChannelStatus> status;
    EXPECT_EQ(hcomm::ChannelProcess::GetChannelsInfo(list, 0, engines, descs, status), HCCL_E_PARA);
}

// --- GetChannelsInfo 正常路径 ---
TEST_F(TestChannelProcess, Ut_GetChannelsInfo_When_Normal_Return_Success)
{
    auto ch = std::make_shared<FakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::READY);
    ChannelHandle handle = RegisterFakeChannel(ch);

    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> descs;
    std::vector<hcomm::ChannelStatus> status;
    EXPECT_EQ(hcomm::ChannelProcess::GetChannelsInfo(&handle, 1, engines, descs, status), HCCL_SUCCESS);
    EXPECT_EQ(engines[0], COMM_ENGINE_AICPU);
    EXPECT_EQ(status[0], hcomm::ChannelStatus::READY);
}

// --- GetChannelsInfo channel not found ---
TEST_F(TestChannelProcess, Ut_GetChannelsInfo_When_NotFound_Return_Error)
{
    ChannelHandle handle = static_cast<ChannelHandle>(0x9999);
    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> descs;
    std::vector<hcomm::ChannelStatus> status;
    EXPECT_NE(hcomm::ChannelProcess::GetChannelsInfo(&handle, 1, engines, descs, status), HCCL_SUCCESS);
}

// --- LaunchAicpuKernel: 全部已 ready 则跳过 ---
TEST_F(TestChannelProcess, Ut_LaunchAicpuKernel_When_AllReady_Skip)
{
    auto ch = std::make_shared<FakeChannel>();
    ch->SetDeviceEntityReady();
    ChannelHandle handle = RegisterFakeChannel(ch);

    HcommChannelDesc descs[1] = {};
    std::vector<int32_t> linkStatus = {hcomm::HCOMM_CHANNEL_STATUS_READY};
    int32_t statusList[1] = {0};
    EXPECT_EQ(AicpuTsChannelHelper::HandleStatus(&handle, 1, COMM_ENGINE_AICPU, descs, linkStatus, statusList), HCCL_SUCCESS);
}

// --- FillAivDevEntities: 全 ready 跳过 ---
TEST_F(TestChannelProcess, Ut_FillAivDevEntities_When_AllReady_Skip)
{
    auto ch = std::make_shared<FakeChannel>();
    ch->SetDeviceEntityReady();
    ChannelHandle handle = RegisterFakeChannel(ch);

    HcommChannelDesc descs[1] = {};
    std::vector<int32_t> linkStatus = {hcomm::HCOMM_CHANNEL_STATUS_READY};
    int32_t statusList[1] = {0};
    EXPECT_EQ(AivChannelHelper::HandleStatus(&handle, 1, descs, linkStatus, statusList), HCCL_SUCCESS);
}

// --- FillAivDevEntities: 非 READY 状态跳过 ---
TEST_F(TestChannelProcess, Ut_FillAivDevEntities_When_NotReadyStatus_Skip)
{
    auto ch = std::make_shared<FakeChannel>();
    ChannelHandle handle = RegisterFakeChannel(ch);

    HcommChannelDesc descs[1] = {};
    std::vector<int32_t> linkStatus = {hcomm::HCOMM_CHANNEL_STATUS_CONNECTING};
    int32_t statusList[1] = {0};
    EXPECT_EQ(AivChannelHelper::HandleStatus(&handle, 1, descs, linkStatus, statusList), HCCL_SUCCESS);
    EXPECT_FALSE(ch->IsDeviceEntityReady());
}

// --- PreAllocChannels: 非 AICPU/AIV engine 直接拷贝 ---
TEST_F(TestChannelProcess, Ut_PreAllocChannels_When_NotSupported_Return_HostHandle)
{
    auto ch = std::make_shared<FakeChannel>(COMM_ENGINE_CPU);
    ChannelHandle target = reinterpret_cast<ChannelHandle>(ch.get());

    ChannelHandle userChannels[1] = {};
    HcommChannelDesc descs[1] = {};
    EXPECT_EQ(hcomm::ChannelProcess::PrepareUserChannels(&target, userChannels, descs, 1, COMM_ENGINE_CPU), HCCL_SUCCESS);
    EXPECT_EQ(userChannels[0], target);
}

// --- PreAllocChannels: 参数校验 ---
TEST_F(TestChannelProcess, Ut_PreAllocChannels_When_Nullptr_Return_E_PTR)
{
    ChannelHandle userChannels[1] = {};
    HcommChannelDesc descs[1] = {};
    EXPECT_EQ(hcomm::ChannelProcess::PrepareUserChannels(nullptr, userChannels, descs, 1, COMM_ENGINE_AICPU), HCCL_E_PTR);
}

// --- PreAllocChannels: channelNum=0 ---
TEST_F(TestChannelProcess, Ut_PreAllocChannels_When_ZeroNum_Return_E_PARA)
{
    ChannelHandle target[1] = {};
    ChannelHandle userChannels[1] = {};
    HcommChannelDesc descs[1] = {};
    EXPECT_EQ(hcomm::ChannelProcess::PrepareUserChannels(target, userChannels, descs, 0, COMM_ENGINE_AICPU), HCCL_E_PARA);
}

// --- UnwrapChannelHandle: handle=0 ---
TEST_F(TestChannelProcess, Ut_UnwrapChannelHandle_When_HandleZero_Return_E_PTR)
{
    ChannelHandle handle = 0;
    EXPECT_EQ(UnwrapChannelHandle(handle), HCCL_E_PTR);
}

// --- UnwrapChannelHandle: magic word 匹配则解包 ---
TEST_F(TestChannelProcess, Ut_UnwrapChannelHandle_When_MagicMatch_Unwrap)
{
    HcommAicpuChannelCtx ctx{};
    ctx.abiHeader.version = HCOMM_AICPU_CHANNEL_CTX_VERSION;
    ctx.abiHeader.magicWord = HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD;
    ctx.deviceChannel = reinterpret_cast<void *>(static_cast<uintptr_t>(0xABCD));

    ChannelHandle handle = reinterpret_cast<ChannelHandle>(&ctx);
    EXPECT_EQ(UnwrapChannelHandle(handle), HCCL_SUCCESS);
    EXPECT_EQ(handle, static_cast<ChannelHandle>(reinterpret_cast<uintptr_t>(ctx.deviceChannel)));
}

// --- UnwrapChannelHandle: magic word 不匹配则保持不变 ---
TEST_F(TestChannelProcess, Ut_UnwrapChannelHandle_When_MagicNotMatch_KeepHandle)
{
    HcommAicpuChannelCtx ctx{};
    ctx.abiHeader.version = 0;
    ctx.abiHeader.magicWord = 0;
    ctx.deviceChannel = reinterpret_cast<void *>(static_cast<uintptr_t>(0xABCD));

    ChannelHandle handle = reinterpret_cast<ChannelHandle>(&ctx);
    ChannelHandle original = handle;
    EXPECT_EQ(UnwrapChannelHandle(handle), HCCL_SUCCESS);
    EXPECT_EQ(handle, original);
}

// 可配置 GetStatus 返回值的 Fake Channel，用于 ChannelGetStatus 测试
class FakeStatusChannel : public hcomm::Channel {
public:
    explicit FakeStatusChannel(hcomm::ChannelStatus status) : status_(status)
    {
    }
    hcomm::HcommChannelKind GetChannelKind() const override
    {
        return hcomm::HcommChannelKind::AICPU_TS_UBOE;
    }
    const HcommChannelDesc &GetChannelDesc() const override
    {
        static HcommChannelDesc defaultDesc{};
        return defaultDesc;
    }
    HcclResult Init() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult GetNotifyNum(uint32_t *notifyNum) const override
    {
        return HCCL_SUCCESS;
    }
    HcclResult GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos) override
    {
        return HCCL_SUCCESS;
    }
    hcomm::ChannelStatus GetStatus() override
    {
        return status_;
    }
    HcclResult Clean() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Resume() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Write(void *dst, const void *src, uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Read(void *dst, const void *src, uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult ChannelFence() override
    {
        return HCCL_SUCCESS;
    }

private:
    hcomm::ChannelStatus status_;
};

// 将 channel 注册到 g_ChannelMap / g_ChannelD2HMap（deviceId=0，handle 即 channel 指针值）
static ChannelHandle RegisterFakeChannel(std::shared_ptr<hcomm::Channel> channel)
{
    ChannelHandle handle = reinterpret_cast<ChannelHandle>(channel.get());
    hcomm::ChannelProcess::g_ChannelMap[handle] = std::move(channel);
    hcomm::DeviceChannelKey key{0, handle};
    hcomm::ChannelProcess::g_ChannelD2HMap[key] = handle;
    return handle;
}

// ChannelGetStatus channelList 空指针测试
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_ChannelListNullptr_Return_HCCL_E_PTR)
{
    int32_t statusList[1] = {0};
    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(nullptr, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ChannelGetStatus statusList 空指针测试
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_StatusListNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle channelList[1] = {0};
    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 全部通道 READY 返回 SUCCESS，并回写状态
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_AllReady_Return_SUCCESS)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::READY));
}

// 存在 FAILED 通道（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasFailed_Return_HCCL_E_NETWORK)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::FAILED));
}

// 存在 SOCKET_TIMEOUT 通道（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasSocketTimeout_Return_HCCL_E_NETWORK)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::SOCKET_TIMEOUT);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::SOCKET_TIMEOUT));
}

// 存在未到终态通道（INIT）返回 HCCL_E_AGAIN
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasNotReady_Return_HCCL_E_AGAIN)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::INIT);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

// READY + FAILED 混合（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_MixReadyAndFailed_Return_HCCL_E_NETWORK)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chFailed = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hFailed = RegisterFakeChannel(chFailed);
    ChannelHandle channelList[2] = {hReady, hFailed};
    int32_t statusList[2] = {0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// READY + INIT 混合（存在未到终态通道）返回 HCCL_E_AGAIN
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_MixReadyAndInit_Return_HCCL_E_AGAIN)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chInit = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::INIT);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hInit = RegisterFakeChannel(chInit);
    ChannelHandle channelList[2] = {hReady, hInit};
    int32_t statusList[2] = {0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

// statusList 已预标记 FAILED 的通道被 continue 跳过，不再获取状态，但计入 failCount
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_StatusListPreMarkedFailed_SkipGetStatus)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chFailed = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hFailed = RegisterFakeChannel(chFailed);
    ChannelHandle channelList[2] = {hReady, hFailed};
    // 第二个通道预标记为 FAILED，触发 continue 分支，不再调用其 GetStatus
    int32_t statusList[2] = {0, static_cast<int32_t>(hcomm::ChannelStatus::FAILED)};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    // 被跳过的通道计入 failCount，故 readyCount(1)+failCount(1)=2 == listNum，且非全 READY，返回 HCCL_E_NETWORK
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    // 被跳过通道的状态保持预标记值不变
    EXPECT_EQ(statusList[1], static_cast<int32_t>(hcomm::ChannelStatus::FAILED));
    // 第一个通道状态被刷新为 READY
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::READY));
}

// handle 未注册时 WithChannelByHandleLocked 返回 HCCL_E_NOT_FOUND
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HandleNotFound_Return_HCCL_E_NOT_FOUND)
{
    ChannelHandle channelList[1] = {static_cast<ChannelHandle>(0xDEADBEEFULL)};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}
