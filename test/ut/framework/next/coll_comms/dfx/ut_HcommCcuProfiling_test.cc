/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <vector>
#include <unordered_map>
#include "ccu_kernel_arg.h"
#include "ccu_kernel_signature.h"
#include "base.h"
#define private public
#include "ccu_task_param_v1.h"
#include "ccu_kernel.h"
#include "ccu_kernel_resource.h"
#include "ccu_rep_context_v1.h"
#include "ccu_task_arg_v1.h"
#include "string_util.h"
#include "task_param.h"
#include "channel_process.h"
#include "hcomm_c_adpt.h"
#include "local_notify_impl.h"
#include "aicpu_launch_manager.h"
#include "llt_hccl_stub_rank_graph.h"
#include "hccl_api_base_test.h"
#include "ccu_kernel_mgr.h"
#include "hcomm_c_adpt.h"
#include "ccu_urma_channel.h"
#include "ccu_comp.h"
#include "ccuTaskException.h"
#include "hcclCommTaskException.h"
#include "ccu_rep_assign_v1.h"
#include "../../../../../src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"

namespace hcomm {

// 模拟CcuUrmaChannel

class MockCcuUrmaChannel {
public:
    MockCcuUrmaChannel(uint16_t channelId = 101, uint32_t dieId = 0) : channelId_(channelId), dieId_(dieId)
    {
    }
    uint16_t GetChannelId()
    {
        return channelId_;
    }
    uint32_t GetDieId()
    {
        return dieId_;
    }

private:
    uint16_t channelId_;
    uint32_t dieId_;
};

// 桩函数。 模拟HcommChannelGet
static MockCcuUrmaChannel *g_mockChannel = nullptr;

HcclResult HcommChannelGet(ChannelHandle channel, void **channelPtr)
{
    if (channelPtr == nullptr)
        return HCCL_E_PTR;
    *channelPtr = g_mockChannel;
    return HCCL_SUCCESS;
}

// 桩函数，模拟SaveDfxTaskInfo
static HcclResult g_saveDfxRet = HCCL_SUCCESS;
HcclResult SaveDfxTaskInfo(const HcclComm comm, const Hccl::TaskParam &taskParam)
{
    return g_saveDfxRet;
}

void SetMockCcuUrmaChannel(MockCcuUrmaChannel *channel)
{
    g_mockChannel = channel;
}
void SetSaveDfxTaskInfoRet(HcclResult ret)
{
    g_saveDfxRet = ret;
}
void ResetStubs()
{
    g_mockChannel = nullptr;
    g_saveDfxRet = HCCL_SUCCESS;
}

// 测试子类
class TestCcuKernel : public CcuKernel {
public:
    using CcuKernel::CcuKernel;
    HcclResult Algorithm() override
    {
        return HCCL_SUCCESS;
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &) override
    {
        return {1, 2, 3, 4};
    }
    // 暴露私有成员用于验证（利用-fno-access-control编译选项，无需修改源码）
    HcclResult GeneTaskParam(const CcuTaskArg &arg, std::vector<CcuTaskParam> &taskParams)
    {
        return HcclResult::HCCL_SUCCESS;
    }

    void SetLoadArgIndex(uint32_t idx)
    {
        loadArgIndex_ = idx;
    }
    void SetMissionId(uint32_t id)
    {
        missionId = id;
    }
    void SetDieId(uint32_t id)
    {
        dieId = id;
    }
    uint32_t GetDieId() const
    {
        return dieId;
    }
    void Work()
    {
        std::shared_ptr<CcuRep::CcuRepBase> rep;
        allLgProfilingReps.push_back(rep);
    }
};

// 测试数据
class MockCcuKernelArg : public hcomm::CcuKernelArg {
public:
    CcuKernelSignature GetKernelSignature() const override
    {
        return CcuKernelSignature{};
    }
};

class CcuKernelTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // 1. 初始化桩函数
        hcomm::ResetStubs();
        // 2. 创建桩函数通道
        mockChannel_ = new MockCcuUrmaChannel(101, 0);
        hcomm::SetMockCcuUrmaChannel(mockChannel_);

        // 3. 构造CcuKernelArg
        kernelArg_.channels = {0x1001, 0x1002};

        // 4. 创建测试实例
        kernel_ = std::make_unique<TestCcuKernel>(kernelArg_);
        kernel_->SetDieId(0);
        kernel_->SetMissionId(100);
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
        // 清理桩函数
        delete mockChannel_;
        hcomm::ResetStubs();
    };
    // 修改 CcuKernelTest 类成员变量
    MockCcuKernelArg kernelArg_;
    std::unique_ptr<TestCcuKernel> kernel_;
    MockCcuUrmaChannel *mockChannel_ = nullptr;

    // 通用测试参数
    const hcomm::GroupInfo testGroupInfo{1, 2, 3};
    const std::vector<ChannelHandle> testChannelsVec{0x1001, 0x1002};
    const ChannelHandle testChannelsArr = 0x1001;
    const HcclDataType testDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    const HcclDataType testOutputDataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    const HcclReduceOp testReduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
};

class Ccukernel_ReportProfilingTest : public hcomm::CcuKernelTest {
protected:
    void SetUp() override
    {
        CcuKernelTest::SetUp();
        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
        // 初始化测试数据
        testThreadHandle = 0x1000;
        testExecId = 0x2000;

        // 有效ProfilingInfo
        validProfInfo.dieId = 1;
        validProfInfo.missionId = 100;
        validProfInfo.instrId = 200;
        validProfInfo.channelId[0] = 101;
        validProfInfo.channelId[1] = 102;

        // 无效channelId的ProfilingInfo
        invalidChannelProfInfo.dieId = 2;
        invalidChannelProfInfo.missionId = 200;
        invalidChannelProfInfo.instrId = 300;
        invalidChannelProfInfo.channelId[0] = INVALID_VALUE_CHANNELID;
    }

    // 测试数据
    ThreadHandle testThreadHandle;
    uint64_t testExecId;
    CcuProfilingInfo validProfInfo;
    CcuProfilingInfo invalidChannelProfInfo;
};

TEST_F(CcuKernelTest, AddCcuProfilingInfo_Normal)
{
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel *channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    void *channelHandle = static_cast<void *>(channel);
    void **handle{nullptr};
    handle = &channelHandle;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    kernel_->Work();
    HcclResult ret = kernel_->AddCcuProfiling(
        testGroupInfo, testChannelsVec, testDataType, testOutputDataType, testReduceOp, "GroupReduce");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuKernelTest, AddProfilingInfo_Normal)
{
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel *channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    void *channelHandle = static_cast<void *>(channel);
    void **handle{nullptr};
    handle = &channelHandle;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    kernel_->Work();
    HcclResult ret
        = kernel_->AddCcuProfiling(&testChannelsArr, 1, testDataType, testOutputDataType, testReduceOp, "GroupReduce");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(Ccukernel_ReportProfilingTest, Ut_HcclReportAivKernel_Normal)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[128] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 6;
    config.hcclRdmaServiceLevel = 0;
    config.hcclRdmaTrafficClass = 0;
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());

    ret = HcclReportAivKernel(comm, 12345);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(Ccukernel_ReportProfilingTest, Ut_HcclReportAicpuKernel_Normal)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommDfx::ReportKernel).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommDfx::GetOpModeFlags)
        .stubs()
        .with(outBound(false), outBound(false))
        .will(returnValue(HCCL_SUCCESS));

    void *commV2 = (void *)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void *)0x1000;
    ;
    char commName[128] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 6;
    config.hcclRdmaServiceLevel = 0;
    config.hcclRdmaTrafficClass = 0;
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void *comm = static_cast<HcclComm>(hcclCommPtr.get());
    char kernelName[128] = "Hello";
    ret = HcclReportAicpuKernel(comm, 12345, kernelName);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuKernelTest, GetCcuProfilingInfo_EmptyCache)
{
    // Ensure profiling cache is empty
    kernel_->GetProfilingInfo().clear();

    std::vector<CcuProfilingInfo> out;
    CcuTaskArg arg{}; // default task arg
    HcclResult ret = kernel_->GetCcuProfilingInfo(arg, out);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(out.empty());
}

TEST_F(CcuKernelTest, GetCcuProfilingInfo_TaskProfilingInfo)
{
    // Prepare a task profiling entry in the kernel's profiling cache
    kernel_->GetProfilingInfo().clear();
    CcuProfilingInfo prof;
    prof.type = static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING);
    prof.name = "UnitTestTaskProfiling";
    prof.dieId = 1;
    prof.missionId = 0; // will be overwritten by GetCcuProfilingInfo
    prof.instrId = 0;   // will be overwritten by GetCcuProfilingInfo

    kernel_->GetProfilingInfo().push_back(prof);

    // Set expected mission and instr ids
    kernel_->SetMissionId(1234);
    kernel_->SetInstrId(4321);

    std::vector<CcuProfilingInfo> out;
    CcuTaskArg arg{};
    HcclResult ret = kernel_->GetCcuProfilingInfo(arg, out);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].missionId, 210);
    EXPECT_EQ(out[0].instrId, 4321u);
    EXPECT_EQ(out[0].name, "UnitTestTaskProfiling");
    EXPECT_EQ(out[0].dieId, 1u);
}

namespace CcuRep {

    class MockCcuRepBase : public CcuRepBase {
    public:
        MockCcuRepBase(CcuRepType type = CcuRepType::BASE, uint16_t startId = 0, uint16_t count = 1)
            : type_(type),
              startId_(startId),
              count_(count)
        {
        }

        bool Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override
        {
            return true;
        }

        std::string Describe() override
        {
            return std::string("MockCcuRepBase");
        }

        uint16_t StartInstrId() const
        {
            return startId_;
        }

        uint16_t InstrCount() override
        {
            return count_;
        }

        CcuRepType Type() const
        {
            return type_;
        }

    private:
        CcuRepType type_;
        uint16_t startId_;
        uint16_t count_;
    };

} // namespace CcuRep

class CcuRepContextTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mockChannel_ = new MockCcuUrmaChannel(101, 0);
        g_mockChannel = mockChannel_;
    }

    void TearDown() override
    {
        delete mockChannel_;
        g_mockChannel = nullptr;
        GlobalMockObject::verify();
    }

    MockCcuUrmaChannel *mockChannel_;
};

TEST_F(CcuRepContextTest, Constructor_Normal)
{
    CcuRep::CcuRepContext context;
    EXPECT_NE(context.CurrentBlock(), nullptr);
}

TEST_F(CcuRepContextTest, CurrentBlock_Normal)
{
    CcuRep::CcuRepContext context;
    auto block = context.CurrentBlock();
    EXPECT_NE(block, nullptr);
}

TEST_F(CcuRepContextTest, SetCurrentBlock_Normal)
{
    CcuRep::CcuRepContext context;
    auto newBlock = std::make_shared<CcuRep::CcuRepBlock>("test_block");
    context.SetCurrentBlock(newBlock);
    EXPECT_EQ(context.CurrentBlock(), newBlock);
}

TEST_F(CcuRepContextTest, Append_Normal)
{
    CcuRep::CcuRepContext context;
    auto rep = std::make_shared<CcuRep::MockCcuRepBase>(CcuRep::CcuRepType::BASE);
    context.Append(rep);
    auto &reps = context.GetRepSequence();
    EXPECT_EQ(reps.size(), 1);
    EXPECT_EQ(reps[0], rep);
}

TEST_F(CcuRepContextTest, GetRepSequence_Empty)
{
    CcuRep::CcuRepContext context;
    auto &reps = context.GetRepSequence();
    EXPECT_EQ(reps.size(), 0);
}

TEST_F(CcuRepContextTest, GetRepByInstrId_NotFound)
{
    CcuRep::CcuRepContext context;
    auto rep = std::make_shared<CcuRep::MockCcuRepBase>(CcuRep::CcuRepType::BASE, 10, 5);
    context.Append(rep);

    auto foundRep = context.GetRepByInstrId(20);
    EXPECT_EQ(foundRep, nullptr);
}

TEST_F(CcuRepContextTest, SetDieId_GetDieId)
{
    CcuRep::CcuRepContext context;
    uint32_t dieId = 5;
    context.SetDieId(dieId);
    EXPECT_EQ(context.GetDieId(), dieId);
}

TEST_F(CcuRepContextTest, SetMissionId_GetMissionId)
{
    CcuRep::CcuRepContext context;
    uint32_t missionId = 100;
    context.SetMissionId(missionId);
    EXPECT_EQ(context.GetMissionId(), missionId);
}

TEST_F(CcuRepContextTest, SetMissionId_OnlyOnce)
{
    CcuRep::CcuRepContext context;
    context.SetMissionId(100);
    context.SetMissionId(200);
    EXPECT_EQ(context.GetMissionId(), 100);
}

TEST_F(CcuRepContextTest, SetMissionKey_GetMissionKey)
{
    CcuRep::CcuRepContext context;
    uint32_t missionKey = 300;
    context.SetMissionKey(missionKey);
    EXPECT_EQ(context.GetMissionKey(), missionKey);
}

TEST_F(CcuRepContextTest, GetProfilingInfo_Empty)
{
    CcuRep::CcuRepContext context;
    auto &profilingInfo = context.GetProfilingInfo();
    EXPECT_EQ(profilingInfo.size(), 0);
}

TEST_F(CcuRepContextTest, GetLGProfilingInfo_Empty)
{
    CcuRep::CcuRepContext context;
    auto &lgProfilingInfo = context.GetLGProfilingInfo();
    EXPECT_EQ(lgProfilingInfo.ccuProfilingInfos.size(), 0);
}

TEST_F(CcuRepContextTest, GetWaiteCkeProfilingReps_Empty)
{
    CcuRep::CcuRepContext context;
    auto &waitReps = context.GetWaiteCkeProfilingReps();
    EXPECT_EQ(waitReps.size(), 0);
}

TEST_F(CcuRepContextTest, CollectProfilingReps_AssignVarToVar)
{
    CcuRep::CcuRepContext context;
    auto rep = std::make_shared<CcuRep::CcuRepAssign>(CcuRep::Variable(&context), CcuRep::Variable(&context));
    rep->subType = CcuRep::AssignSubType::VAR_TO_VAR;
    context.CollectProfilingReps(rep);

    auto &profilingInfo = context.GetLGProfilingInfo();
    EXPECT_EQ(profilingInfo.assignProfilingReps.size(), 1);
}

TEST_F(CcuRepContextTest, AddSqeProfiling_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    context.AddSqeProfiling();

    auto &profilingInfo = context.GetProfilingInfo();
    EXPECT_EQ(profilingInfo.size(), 1);
    EXPECT_EQ(profilingInfo[0].type, (uint8_t)CcuProfilinType::CCU_TASK_PROFILING);
    EXPECT_EQ(profilingInfo[0].name, "CCU_KERNEL");
    EXPECT_EQ(profilingInfo[0].dieId, 1);
}

TEST_F(CcuRepContextTest, AddProfiling_WithNameAndMask)
{
    CcuRep::CcuRepContext context;
    std::string name = "TestProfiling";
    uint32_t mask = 0xFF;
    int32_t ret = context.AddProfiling(name, mask);

    EXPECT_EQ(ret, 0);
    auto &profilingInfo = context.GetProfilingInfo();
    EXPECT_EQ(profilingInfo.size(), 1);
    EXPECT_EQ(profilingInfo[0].name, name);
    EXPECT_EQ(profilingInfo[0].mask, mask);
}

TEST_F(CcuRepContextTest, AddProfiling_WithChannelsNullPtr)
{
    CcuRep::CcuRepContext context;
    int32_t ret = context.AddProfiling(nullptr, 2);
    EXPECT_NE(ret, 0);
}
// ========== GenErrorInfo系列函数测试 ==========

class GenErrorInfoTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(HccpRaCustomChannel)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        baseInfo_.dieId = 0;
        baseInfo_.deviceId = 0;
        baseInfo_.missionId = 1;
        baseInfo_.currentInsId = 10;
        baseInfo_.status = 0x0101;
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
    ErrorInfoBase baseInfo_;
};

class MockTaskCcuRepBase : public CcuRep::CcuRepBase {
public:
    MockTaskCcuRepBase(CcuRep::CcuRepType type, uint16_t instrId = 10)
    {
        this->type = type;
        this->instrId = instrId;
    }
    ~MockTaskCcuRepBase() override = default;
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "MockRep";
    }
    CcuRep::CcuRepType GetType() const
    {
        return type;
    }
};

class MockCcuRepLocRecordEvent : public CcuRep::CcuRepBase {
public:
    MockCcuRepLocRecordEvent(uint16_t eventId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::LOC_RECORD_EVENT;
        this->instrId = 10;
        eventId_ = eventId;
        mask_ = mask;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "LocRecordEvent";
    }
    uint16_t GetEventId()
    {
        return eventId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }

private:
    uint16_t eventId_;
    uint32_t mask_;
};

class MockCcuRepLocWaitEvent : public CcuRep::CcuRepBase {
public:
    MockCcuRepLocWaitEvent(uint16_t eventId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::LOC_WAIT_EVENT;
        this->instrId = 11;
        eventId_ = eventId;
        mask_ = mask;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "LocWaitEvent";
    }
    uint16_t GetEventId()
    {
        return eventId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }

private:
    uint16_t eventId_;
    uint32_t mask_;
};

class MockCcuRepRemPostSem : public CcuRep::CcuRepBase {
public:
    MockCcuRepRemPostSem(uint32_t id, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::REM_POST_SEM;
        this->instrId = 13;
        id_ = id;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "RemPostSem";
    }
    uint32_t GetId()
    {
        return id_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint32_t GetChannelId()
    {
        return channelId_;
    }

private:
    uint32_t id_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepRemWaitSem : public CcuRep::CcuRepBase {
public:
    MockCcuRepRemWaitSem(uint32_t id, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::REM_WAIT_SEM;
        this->instrId = 14;
        id_ = id;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "RemWaitSem";
    }
    uint32_t GetId()
    {
        return id_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint32_t GetChannelId()
    {
        return channelId_;
    }

private:
    uint32_t id_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepRemPostVar : public CcuRep::CcuRepBase {
public:
    MockCcuRepRemPostVar(
        uint32_t rmtCkeId, uint32_t rmtXnId, uint32_t paramIndex, uint32_t semIndex, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::REM_POST_VAR;
        this->instrId = 15;
        rmtCkeId_ = rmtCkeId;
        rmtXnId_ = rmtXnId;
        paramIndex_ = paramIndex;
        semIndex_ = semIndex;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "RemPostVar";
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint32_t GetRmtCkeId()
    {
        return rmtCkeId_;
    }
    uint32_t GetRmtXnId()
    {
        return rmtXnId_;
    }
    uint32_t GetParamIndex()
    {
        return paramIndex_;
    }
    CcuRep::Variable GetParam()
    {
        CcuRep::Variable v;
        v.Reset(100);
        return v;
    }
    uint32_t GetChannelId()
    {
        return channelId_;
    }

private:
    uint32_t rmtCkeId_;
    uint32_t rmtXnId_;
    uint32_t paramIndex_;
    uint32_t semIndex_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepRecordSharedNotify : public CcuRep::CcuRepBase {
public:
    MockCcuRepRecordSharedNotify(uint16_t notifyId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::RECORD_SHARED_NOTIFY;
        this->instrId = 16;
        notifyId_ = notifyId;
        mask_ = mask;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "RecordSharedNotify";
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint16_t GetNotifyId()
    {
        return notifyId_;
    }

private:
    uint16_t notifyId_;
    uint32_t mask_;
};

class MockCcuRepRead : public CcuRep::CcuRepBase {
public:
    MockCcuRepRead(uint16_t locAddrId, uint16_t locTokenId, uint16_t remAddrId, uint16_t remTokenId, uint16_t lenId,
        uint16_t semId, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::READ;
        this->instrId = 17;
        locAddrId_ = locAddrId;
        locTokenId_ = locTokenId;
        remAddrId_ = remAddrId;
        remTokenId_ = remTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "Read";
    }
    uint16_t GetLocAddrId()
    {
        return locAddrId_;
    }
    uint16_t GetLocTokenId()
    {
        return locTokenId_;
    }
    uint16_t GetRemAddrId()
    {
        return remAddrId_;
    }
    uint16_t GetRemTokenId()
    {
        return remTokenId_;
    }
    uint16_t GetLenId()
    {
        return lenId_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint32_t GetChannelId()
    {
        return channelId_;
    }

private:
    uint16_t locAddrId_;
    uint16_t locTokenId_;
    uint16_t remAddrId_;
    uint16_t remTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepWrite : public CcuRep::CcuRepBase {
public:
    MockCcuRepWrite(uint16_t locAddrId, uint16_t locTokenId, uint16_t remAddrId, uint16_t remTokenId, uint16_t lenId,
        uint16_t semId, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::WRITE;
        this->instrId = 18;
        locAddrId_ = locAddrId;
        locTokenId_ = locTokenId;
        remAddrId_ = remAddrId;
        remTokenId_ = remTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "Write";
    }
    uint16_t GetLocAddrId()
    {
        return locAddrId_;
    }
    uint16_t GetLocTokenId()
    {
        return locTokenId_;
    }
    uint16_t GetRemAddrId()
    {
        return remAddrId_;
    }
    uint16_t GetRemTokenId()
    {
        return remTokenId_;
    }
    uint16_t GetLenId()
    {
        return lenId_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint32_t GetChannelId()
    {
        return channelId_;
    }

private:
    uint16_t locAddrId_;
    uint16_t locTokenId_;
    uint16_t remAddrId_;
    uint16_t remTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepLocalCpy : public CcuRep::CcuRepBase {
public:
    MockCcuRepLocalCpy(uint16_t srcAddrId, uint16_t srcTokenId, uint16_t dstAddrId, uint16_t dstTokenId, uint16_t lenId,
        uint16_t semId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::LOCAL_CPY;
        this->instrId = 19;
        srcAddrId_ = srcAddrId;
        srcTokenId_ = srcTokenId;
        dstAddrId_ = dstAddrId;
        dstTokenId_ = dstTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "LocalCpy";
    }
    uint16_t GetSrcAddrId()
    {
        return srcAddrId_;
    }
    uint16_t GetSrcTokenId()
    {
        return srcTokenId_;
    }
    uint16_t GetDstAddrId()
    {
        return dstAddrId_;
    }
    uint16_t GetDstTokenId()
    {
        return dstTokenId_;
    }
    uint16_t GetLenId()
    {
        return lenId_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }

private:
    uint16_t srcAddrId_;
    uint16_t srcTokenId_;
    uint16_t dstAddrId_;
    uint16_t dstTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
};

class MockCcuRepLocalReduce : public CcuRep::CcuRepBase {
public:
    MockCcuRepLocalReduce(uint16_t srcAddrId, uint16_t srcTokenId, uint16_t dstAddrId, uint16_t dstTokenId,
        uint16_t lenId, uint16_t semId, uint32_t mask, uint16_t dataType, uint16_t opType)
    {
        this->type = CcuRep::CcuRepType::LOCAL_REDUCE;
        this->instrId = 20;
        srcAddrId_ = srcAddrId;
        srcTokenId_ = srcTokenId;
        dstAddrId_ = dstAddrId;
        dstTokenId_ = dstTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        dataType_ = dataType;
        opType_ = opType;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "LocalReduce";
    }
    uint16_t GetSrcAddrId()
    {
        return srcAddrId_;
    }
    uint16_t GetSrcTokenId()
    {
        return srcTokenId_;
    }
    uint16_t GetDstAddrId()
    {
        return dstAddrId_;
    }
    uint16_t GetDstTokenId()
    {
        return dstTokenId_;
    }
    uint16_t GetLenId()
    {
        return lenId_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint16_t GetDataType()
    {
        return dataType_;
    }
    uint16_t GetOpType()
    {
        return opType_;
    }

private:
    uint16_t srcAddrId_;
    uint16_t srcTokenId_;
    uint16_t dstAddrId_;
    uint16_t dstTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint16_t dataType_;
    uint16_t opType_;
};

class MockCcuRepBufRead : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufRead(uint16_t dstAddrId, uint16_t srcAddrId, uint16_t srcTokenId, uint16_t lenId, uint16_t semId,
        uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::BUF_READ;
        this->instrId = 21;
        dstAddrId_ = dstAddrId;
        srcAddrId_ = srcAddrId;
        srcTokenId_ = srcTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "BufRead";
    }
    uint16_t GetDstAddrId()
    {
        return dstAddrId_;
    }
    uint16_t GetSrcAddrId()
    {
        return srcAddrId_;
    }
    uint16_t GetSrcTokenId()
    {
        return srcTokenId_;
    }
    uint16_t GetLenId()
    {
        return lenId_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint32_t GetChannelId()
    {
        return channelId_;
    }

private:
    uint16_t dstAddrId_;
    uint16_t srcAddrId_;
    uint16_t srcTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepBufWrite : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufWrite(uint16_t srcId, uint16_t dstAddrId, uint16_t dstTokenId, uint16_t lenId, uint16_t semId,
        uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::BUF_WRITE;
        this->instrId = 22;
        srcId_ = srcId;
        dstAddrId_ = dstAddrId;
        dstTokenId_ = dstTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "BufWrite";
    }
    uint16_t GetSrcId()
    {
        return srcId_;
    }
    uint16_t GetDstAddrId()
    {
        return dstAddrId_;
    }
    uint16_t GetDstTokenId()
    {
        return dstTokenId_;
    }
    uint16_t GetLenId()
    {
        return lenId_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint32_t GetChannelId()
    {
        return channelId_;
    }

private:
    uint16_t srcId_;
    uint16_t dstAddrId_;
    uint16_t dstTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepBufLocRead : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufLocRead(
        uint16_t srcAddrId, uint16_t srcTokenId, uint16_t dstId, uint16_t lenId, uint16_t semId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::BUF_LOC_READ;
        this->instrId = 23;
        srcAddrId_ = srcAddrId;
        srcTokenId_ = srcTokenId;
        dstId_ = dstId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "BufLocRead";
    }
    uint16_t GetSrcAddrId()
    {
        return srcAddrId_;
    }
    uint16_t GetSrcTokenId()
    {
        return srcTokenId_;
    }
    uint16_t GetDstId()
    {
        return dstId_;
    }
    uint16_t GetLenId()
    {
        return lenId_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }

private:
    uint16_t srcAddrId_;
    uint16_t srcTokenId_;
    uint16_t dstId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
};

class MockCcuRepBufLocWrite : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufLocWrite(
        uint16_t srcId, uint16_t dstAddrId, uint16_t dstTokenId, uint16_t lenId, uint16_t semId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::BUF_LOC_WRITE;
        this->instrId = 24;
        srcId_ = srcId;
        dstAddrId_ = dstAddrId;
        dstTokenId_ = dstTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "BufLocWrite";
    }
    uint16_t GetSrcAddrId()
    {
        return srcId_;
    }
    uint16_t GetDstTokenId()
    {
        return dstTokenId_;
    }
    uint16_t GetDstAddrId()
    {
        return dstAddrId_;
    }
    uint16_t GetLenId()
    {
        return lenId_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }

private:
    uint16_t srcId_;
    uint16_t dstAddrId_;
    uint16_t dstTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
};

class MockCcuRepBufReduce : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufReduce(uint16_t count, uint16_t dataType, uint16_t outputDataType, uint16_t opType, uint16_t semId,
        uint32_t mask, const std::vector<CcuRep::CcuBuf> &mem, uint16_t xnLengthId)
    {
        this->type = CcuRep::CcuRepType::BUF_REDUCE;
        this->instrId = 25;
        count_ = count;
        dataType_ = dataType;
        outputDataType_ = outputDataType;
        opType_ = opType;
        semId_ = semId;
        mask_ = mask;
        mem_ = mem;
        xnLengthId_ = xnLengthId;
    }
    bool Translate(CcuInstr *&instr, uint16_t &instrId, const CcuRep::TransDep &dep) override
    {
        return true;
    }
    std::string Describe() override
    {
        return "BufReduce";
    }
    const std::vector<CcuRep::CcuBuf> &GetMem()
    {
        return mem_;
    }
    uint16_t GetCount()
    {
        return count_;
    }
    uint16_t GetDataType()
    {
        return dataType_;
    }
    uint16_t GetOutputDataType()
    {
        return outputDataType_;
    }
    uint16_t GetOpType()
    {
        return opType_;
    }
    uint16_t GetXnLengthId()
    {
        return xnLengthId_;
    }
    uint32_t GetMask()
    {
        return mask_;
    }
    uint16_t GetSemId()
    {
        return semId_;
    }

private:
    uint16_t count_;
    uint16_t dataType_;
    uint16_t outputDataType_;
    uint16_t opType_;
    uint16_t semId_;
    uint32_t mask_;
    std::vector<CcuRep::CcuBuf> mem_;
    uint16_t xnLengthId_;
};

TEST_F(CcuRepContextTest, AddProfilingnameandmask_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    std::string name = "CCU_KERNEL";
    uint32_t mask = 0;
    int ret = context.AddProfiling(name, mask);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(CcuRepContextTest, AddProfilingchannel_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    std::string name = "CCU_KERNEL";
    uint32_t mask = 0;
    uint32_t signalIndex = 0;
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel *channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(channel);
    void **handle{nullptr};
    void *voidHandle = reinterpret_cast<void *>(channelHandle);
    handle = &voidHandle;
    uint32_t locCkeId = 1;
    uint32_t channelid = 2;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetLocCkeByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(locCkeId))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(channelid));

    int ret = context.AddProfiling(channelHandle, name, signalIndex, mask);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(CcuRepContextTest, AddProfilingchannelNum_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    std::string name = "CCU_KERNEL";
    uint32_t num = 1;
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel *channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(channel);
    void **handle{nullptr};
    void *voidHandle = reinterpret_cast<void *>(channelHandle);
    handle = &voidHandle;
    uint32_t channelid = 2;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(channelid));

    int ret = context.AddProfiling(&channelHandle, num);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(CcuRepContextTest, AddProfilingchannelNumMuch_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    std::string name = "CCU_KERNEL";
    uint32_t num = 1;
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel *channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(channel);
    void **handle{nullptr};
    void *voidHandle = reinterpret_cast<void *>(channelHandle);
    handle = &voidHandle;
    uint32_t channelid = 2;
    uint32_t missonid = 2;
    HcommDataType hcommDataType = HcommDataType::HCOMM_DATA_TYPE_INT8;
    HcommDataType hcommOutputDataType = HcommDataType::HCOMM_DATA_TYPE_INT8;
    HcommReduceOp hcommOpType = HcommReduceOp::HCOMM_REDUCE_SUM;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(channelid));
    MOCKER_CPP(&CcuRepContext::GetMissionId).stubs().will(returnValue(missonid));
    std::shared_ptr<CcuRepBase> baserep;
    context.allLgProfilingReps.push_back(baserep);
    int ret = context.AddProfiling(&channelHandle, num, hcommDataType, hcommOutputDataType, hcommOpType);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

namespace {

    class MockHccpRaCustomChannel {
    public:
        static uint32_t dieId_;
        static uint32_t callCount_;
        static uint32_t lastOpCode_;

        static void Setup()
        {
            callCount_ = 0;
            lastOpCode_ = 0;
        }
    };

    uint32_t MockHccpRaCustomChannel::dieId_ = 0;
    uint32_t MockHccpRaCustomChannel::callCount_ = 0;
    uint32_t MockHccpRaCustomChannel::lastOpCode_ = 0;

} // namespace

class CcuComponentTest : public ::testing::Test {
protected:
    void SetUp() override
    {
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(CcuComponentTest, CcuCleanTaskKillState_InvalidParam)
{
    auto ret = CcuComponent::GetInstance(0).CcuCleanTaskKillState(-1);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);

    ret = CcuComponent::GetInstance(0).CcuCleanTaskKillState(256);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CcuComponentTest, CleanDieCkes_InvalidDieId)
{
    constexpr uint8_t MAX_IODIE_NUM = 2;
    auto ret = CcuComponent::GetInstance(0).CleanDieCkes(MAX_IODIE_NUM);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CcuComponentTest, CcuComponentTestSetProcess)
{
    CcuOpcodeType opCode = CcuOpcodeType::CCU_U_OP_CLEAN_TASKKILL_STATE;
    // 模拟HccpRaCustomChannel
    MOCKER(HccpRaCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    auto ret = CcuComponent::GetInstance(0).SetProcess(opCode);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, CcuComponentSetTaskKillDone)
{
    MOCKER_CPP(&CcuComponent::SetProcess).stubs().will(returnValue(HCCL_SUCCESS));
    CcuComponent::GetInstance(0).status = CcuComponent::GetInstance(0).CcuTaskKillStatus::TASK_KILL;
    auto ret = CcuComponent::GetInstance(0).SetTaskKillDone();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, CcuSetTaskKillDonefailed)
{
    auto ret = CcuComponent::GetInstance(0).CcuSetTaskKillDone(-1);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CcuComponentTest, CleanDiId)
{
    constexpr uint8_t IODIE = 0;
    uint32_t ckeNum = 1;
    MOCKER_CPP(&CcuResSpecifications::GetCkeNum)
        .stubs()
        .with(mockcpp::any(), outBound(ckeNum))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CcuComponent::GetInstance(0).dieEnableFlags_[0] = true;
    auto ret = CcuComponent::GetInstance(0).CleanDieCkes(IODIE);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

} // namespace hcomm

#undef private