/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "hccl_api_base_test.h"

#define private public
#define protected public

#include "log.h"
#include "adapter_rts.h"
#include "hcomm_c_adpt.h"

#include "ccu_device_pub.h"
#include "ccu_kernel_mgr.h"
#include "ccu_instance_mgr.h"
#include "ccu_device_res.h"
#include "ccu_launch.h"

#include "mocks/ccu_device_mock_utils.h"
#include "mocks/ccu_channel_mock_utils.h"

#include "adapter_rts.h"
#include "adapter_hal_pub.h"

#include "hcomm_primitives.h"

#include "ccu_kernel_impl/ccu_var_add_simple_demo.h"
#include "ccu_kernel_impl/ccu_loop_add_demo.h"
#include "ccu_kernel_impl/ccu_func_call_demo.h"
#include "ccu_kernel_impl/ccu_jump_demo.h"
#include "ccu_kernel_impl/ccu_reduce_scatter_mesh1d_demo.h"
#include "ccu_kernel_impl/ccu_groupcopy_demo.h"

#undef protected
#undef private

class HcommCcuControlApiTest : public BaseInit {
public:
    void SetUp() override {
        GlobalMockObject::verify();
        BaseInit::SetUp();
        // 将enableEntryLog默认返回为true
        MOCKER(GetExternalInputHcclEnableEntryLog)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(true));
    }
    void TearDown() override {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
protected:
};

static std::pair<EndpointHandle, ChannelHandle> MockCcuChannelConnect(
    uint32_t srcDevPhyId, uint32_t dstDevPhyId,
    uint32_t srcIp, uint32_t dstIp, CommEngine commEngine)
{
    HcommResult hcommRet = 0;
    CcuResult ccuRest = CcuResult::CCU_E_RESERVED;

    CommAddr srcAddr{}, dstAddr{};
    srcAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    srcAddr.addr.s_addr = srcIp;
    dstAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    dstAddr.addr.s_addr = dstIp;

    const auto &srcEpDesc = MockEndpointDesc(srcAddr, srcDevPhyId);
    EndpointHandle srcEpHandle{};
    hcommRet = HcommEndpointCreate(&srcEpDesc, &srcEpHandle);
    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));

    const auto &dstEpDesc = MockEndpointDesc(dstAddr, dstDevPhyId);

    const auto &socket = MockHcclSocket(srcAddr, dstAddr);
    HcommSocket socketPtr = static_cast<HcommSocket>(socket.get());
    const auto &rmaBuffer = MockUbRmaBuffer();
    void *memHandle = static_cast<void *>(rmaBuffer.get());
    auto channelDesc = MockHcommChannelDesc(dstEpDesc, socketPtr, memHandle);
    constexpr uint32_t channelNum = 1;
    ChannelHandle channelHandle{0};
    hcommRet = HcommChannelCreate(srcEpHandle, commEngine, &channelDesc, channelNum, &channelHandle);
    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));

    int32_t statusList[] = {0};

    constexpr uint32_t MAX_LOOP_TIME = 100;
    uint32_t leftTime = MAX_LOOP_TIME;
    while (leftTime--) {
        hcommRet = HcommChannelGetStatus(&channelHandle, channelNum, statusList);
        if (hcommRet == static_cast<HcommResult>(HcclResult::HCCL_SUCCESS)) {
            break;
        }

        if (hcommRet != static_cast<HcommResult>(HcclResult::HCCL_E_AGAIN)) {
            HCCL_ERROR("[%s] invalid ret[%d].", __func__, hcommRet);
            break;
        }
    }

    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));
    return {srcEpHandle, channelHandle};
}

static void MockChannelDestory(const std::pair<EndpointHandle, ChannelHandle> &handles)
{
    HcommResult hcommRet = 0;
    constexpr uint32_t channelNum = 1;
    hcommRet = HcommChannelDestroy(&(handles.second), channelNum);
    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));

    hcommRet = HcommEndpointDestroy(handles.first);
    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));
}

static ThreadHandle MockThreadAllocWithStream(CommEngine commEngine)
{
    // 选择调用Hcomm接口，而不是对具体的Thread实现打桩，减少内部依赖
    MOCKER(aclrtStreamGetId).stubs().will(returnValue(0));
    bool devRunning = false;
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(devRunning))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    
    constexpr uint32_t fakeNotifyNum = 0;
    aclrtStream fakeStream{(void *)0x12345678};
    ThreadHandle fakeThreadHandle{};
    EXPECT_EQ(HcommThreadAllocWithStream(commEngine, fakeStream,
        fakeNotifyNum, &fakeThreadHandle), HcclResult::HCCL_SUCCESS);

    return fakeThreadHandle;
}

#define CCU_FUNC_KERNEL_TEST(testName, demoFunc, expectRegisterSuccess)                                      \
TEST_F(HcommCcuControlApiTest, testName)                                                                    \
{                                                                                                           \
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;                                                           \
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;                                               \
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));                                     \
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);                                            \
    MOCKER(hrtGetDevice).stubs()                                                                            \
        .with(outBoundP(&fakeDeviceLogicId))                                                                \
        .will(returnValue(HcclResult::HCCL_SUCCESS));                                                       \
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()                                                               \
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())                             \
        .will(returnValue(HcclResult::HCCL_SUCCESS));                                                       \
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;                                 \
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);                                                         \
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);        \
    MockCcuChannelGetRes();                                                                                 \
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));                                  \
                                                                                                            \
    CcuResDesc resDesc{};                                                                                   \
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;                                                              \
    resDesc.insType = CcuInstanceType::CCU_MS;                                                              \
    constexpr uint32_t descNum = 1;                                                                         \
    CcuInsHandle insHandle{0};                                                                              \
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);                         \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
                                                                                                            \
    ccuRet = HcommCcuKernelRegisterStart(insHandle);                                                        \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
                                                                                                            \
    int32_t dummyArg = 0;                                                                                   \
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);                                          \
    const void *kernelArgs[] = {kernelArg};                                                                 \
    CcuKernelHandle kernelHandle{0};                                                                        \
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);                                                   \
    constexpr uint32_t fakeDieId = 0;                                                                       \
    constexpr uint32_t kernelArgNum = 1;                                                                    \
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId, const_cast<char *>(#demoFunc),                    \
        kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);                                               \
    if (expectRegisterSuccess) {                                                                            \
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                          \
        ccuRet = HcommCcuKernelRegisterEnd(insHandle);                                                      \
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                          \
    } else {                                                                                                \
        EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);                                                          \
    }                                                                                                       \
                                                                                                            \
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);                                                                 \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
}

CCU_FUNC_KERNEL_TEST(Ut_FuncCallBasic_Expect_Success, CcuFuncCallBasicDemoKernel, true)
CCU_FUNC_KERNEL_TEST(Ut_FuncCallReuse_Expect_Success, CcuFuncCallReuseDemoKernel, true)
CCU_FUNC_KERNEL_TEST(Ut_FuncCallMultiArg_Expect_Success, CcuFuncCallMultiArgDemoKernel, true)
CCU_FUNC_KERNEL_TEST(Ut_FuncCallInLoop_Expect_Fail, CcuFuncCallInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_FuncCallNested_Expect_Fail, CcuFuncCallNestedInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_IfInLoop_Expect_Fail, CcuIfInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_NotifyRecordInLoop_Expect_Fail, CcuNotifyRecordInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_WriteVarWithNotifyInLoop_Expect_Fail, CcuWriteVarWithNotifyInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_EventRecordTagInLoop_Expect_Fail, CcuEventRecordTagInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_EventRecordInLoop_Expect_Fail, CcuEventRecordInLoopInvalidDemoKernel, false)

TEST_F(HcommCcuControlApiTest, Ut_LoopObjectApi_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = CcuInstanceType::CCU_MS;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuLoopAddKernelArg demoArg{};
    demoArg.numA = 3;
    demoArg.numB = 4;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void *>(CcuLoopAddDemoKernel);

    char *kernelFuncName = "CcuLoopAddDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRegister_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    // 整体打桩，处理ccu资源
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId); // 先处理网络设备，再初始化ccu
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    // ccuInstance构建（正常在通信域创建中，本用例仅测试hcomm接口）
    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;

    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // 建链流程打桩
    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383; // 需要与RaGetDevEidInfoList接口打桩一致
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    // 构造CcuKernel实现
    auto demoFunc = CcuLoadStoreDemoKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.numA = 1;
    demoArg.numB = 2;
    demoArg.channelHandle = handlePair.second;

    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    // 重置CCU资源
    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // kernel注册
    char *kernelFuncName = "ccu_var_add_simple_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // kernel翻译
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    
    // 申请流，假定已经获取了threadHandle
    auto fakeThreadHandle = MockThreadAllocWithStream(commEngine);

    // kernel下发
    // 需要与样例需要的load args对应
    std::vector<uint64_t> taskArgs{};
    void *fakeTaskArgs = static_cast<void *>(taskArgs.data());
    uint32_t fakeArgNum = taskArgs.size();
    EXPECT_EQ(HcommCcuKernelLaunch(fakeThreadHandle, kernelHandle,
        fakeTaskArgs, fakeArgNum), CcuResult::CCU_SUCCESS);

    // 清理各种资源，析构有时序要求
    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelDoWhile_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 3;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuDoWhileWhileDemoKernel;
    CcuDoWhileWhileDemoKernelArg demoArg{};
    demoArg.loopCount = 5;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_do_while_while_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedIfOuterElse_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 4;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuNestedIfOuterElseDemoKernel;
    CcuNestedIfOuterElseDemoKernelArg demoArg{};
    demoArg.outerVal = 1;
    demoArg.outerExpected = 1;
    demoArg.innerVal = 2;
    demoArg.innerExpected = 2;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_nested_if_outer_else_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedIfInnerElse_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 5;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuNestedIfInnerElseDemoKernel;
    CcuNestedIfInnerElseDemoKernelArg demoArg{};
    demoArg.outerVal = 1;
    demoArg.outerExpected = 1;
    demoArg.innerVal = 2;
    demoArg.innerExpected = 2;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_nested_if_inner_else_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelDoWhileUnified_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 6;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuDoWhileUnifiedDemoKernel;
    CcuDoWhileUnifiedDemoKernelArg demoArg{};
    demoArg.loopCount = 5;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_do_while_unified_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_DoWhileLabelStackPopForWhile_When_Adjacent_Expect_ReturnLabel)
{
    hcomm::CcuKernel kernel;
    const char *label = "ut_dw_adjacent";

    EXPECT_EQ(kernel.DoWhileBegin(label), CcuResult::CCU_SUCCESS);
    kernel.DoWhileLabelStackPush(label);

    const char *popped = kernel.DoWhileLabelStackPopForWhile();
    ASSERT_NE(popped, nullptr);
    EXPECT_STREQ(popped, label);
    EXPECT_TRUE(kernel.doWhileLabelStack_.empty());
}

TEST_F(HcommCcuControlApiTest, Ut_DoWhileLabelStackPopForWhile_When_DanglingAppend_Expect_ReturnNullptr)
{
    hcomm::CcuKernel kernel;
    const char *labelOuter = "ut_dw_outer";
    const char *labelDangling = "ut_dw_dangling";

    EXPECT_EQ(kernel.DoWhileBegin(labelOuter), CcuResult::CCU_SUCCESS);
    kernel.DoWhileLabelStackPush(labelOuter);

    // 用第二次 DoWhileBegin 模拟 CCU_DO 与 CCU_WHILE 之间夹杂了会 Append 指令的代码。
    EXPECT_EQ(kernel.DoWhileBegin(labelDangling), CcuResult::CCU_SUCCESS);

    // 快照 rep 数与当前不匹配，PopForWhile 返回 nullptr 拒绝配对，但仍需弹出栈条目。
    const char *popped = kernel.DoWhileLabelStackPopForWhile();
    EXPECT_EQ(popped, nullptr);
    EXPECT_TRUE(kernel.doWhileLabelStack_.empty());
}

TEST_F(HcommCcuControlApiTest, Ut_DoWhileLabelStackPopForWhile_When_InLoopBody_Expect_ReturnNullptr)
{
    // loop body 内(CurrentBlock 为 LOOP_BLOCK)PopForWhile 应直接返回 nullptr 且不消费栈条目。
    hcomm::CcuKernel kernel;
    CcuLoop loop = 0;
    ASSERT_EQ(kernel.LoopCreate(&loop), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(kernel.LoopBodyEnter(loop), CcuResult::CCU_SUCCESS);

    const char *label = "ut_dw_in_loop";
    kernel.DoWhileLabelStackPush(label);
    ASSERT_FALSE(kernel.doWhileLabelStack_.empty());

    EXPECT_EQ(kernel.DoWhileLabelStackPopForWhile(), nullptr);
    EXPECT_FALSE(kernel.doWhileLabelStack_.empty());

    EXPECT_EQ(kernel.LoopBodyExit(loop), CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelLoopAdd_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 7;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuLoopAddDemoKernel;
    CcuLoopAddKernelArg demoArg{};
    demoArg.numA = 7;
    demoArg.numB = 11;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_loop_add_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRemoteRead_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 8;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuRemoteReadKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.channelHandle = handlePair.second;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_remote_read_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRemoteWrite_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 9;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuRemoteWriteKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.channelHandle = handlePair.second;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_remote_write_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelAlloc_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuAllocDemoKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.channelHandle = handlePair.second;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_alloc_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelReduceScatterMesh1d_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    // 整体打桩，处理ccu资源
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId); // 先处理网络设备，再初始化ccu
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    // ccuInstance构建（正常在通信域创建中，本用例仅测试hcomm接口）
    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;

    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // 建链流程打桩
    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383; // 需要与RaGetDevEidInfoList接口打桩一致
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    // 构造CcuKernel实现
    auto demoFunc = CcuReduceScatterMesh1dKernel;
    ReduceScatterKernelArg demoArg{};
    demoArg.rankSize       = 2;
    demoArg.rankId         = 0;
    demoArg.channelCount   = 1;
    demoArg.channels[0]    = handlePair.second;
    demoArg.dataType       = HCCL_DATA_TYPE_FP16;
    demoArg.outputDataType = HCCL_DATA_TYPE_FP16;
    demoArg.reduceOp       = HCCL_REDUCE_SUM;

    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    // 重置CCU资源
    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // kernel注册
    char *kernelFuncName = "ccu_reduce_scatter_mesh1d_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // kernel翻译
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    
    // 申请流，假定已经获取了threadHandle
    auto fakeThreadHandle = MockThreadAllocWithStream(commEngine);

    // kernel下发
    std::vector<uint64_t> taskArgs(15, 0);
    void *fakeTaskArgs = static_cast<void *>(taskArgs.data());
    uint32_t fakeArgSize = taskArgs.size();
    EXPECT_EQ(HcommCcuKernelLaunch(fakeThreadHandle, kernelHandle,
        fakeTaskArgs, fakeArgSize), CcuResult::CCU_SUCCESS);

    // 清理各种资源，析构有时序要求
    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedInIfIf_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 11;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuNestedInIfIfDemoKernel;
    CcuNestedInIfIfDemoKernelArg demoArg{};
    demoArg.outerVal = 1;
    demoArg.outerExpected = 1;
    demoArg.innerVal_1 = 2;
    demoArg.innerExpected_1 = 2;
    demoArg.innerVal_2 = 3;
    demoArg.innerExpected_2 = 3;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_nested_in_if_if_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelAllGatherMesh1dMem2Mem_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    HcommResult hcclRet = 0;
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 12;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = MS_INS_TPYE;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    // 构造 demo 入参：rankSize=2, rankId=0, 1 个 peer 通道。
    auto demoFunc = CcuAllGatherMesh1dMem2MemKernel;
    AllGatherKernelArg demoArg{};
    demoArg.rankSize     = 2;
    demoArg.rankId       = 0;
    demoArg.channelCount = 1;
    demoArg.channels[0]  = handlePair.second;

    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg  = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_all_gather_mesh1d_mem2mem_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    auto fakeThreadHandle = MockThreadAllocWithStream(commEngine);

    std::vector<uint64_t> taskArgs(15, 0);
    void *fakeTaskArgs = static_cast<void *>(taskArgs.data());
    uint32_t fakeArgSize = taskArgs.size();
    EXPECT_EQ(HcommCcuKernelLaunch(fakeThreadHandle, kernelHandle,
        fakeTaskArgs, fakeArgSize), CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle, fakeDeviceLogicId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

// HcommCcuInsDestroy 接口在线程 DeviceId 与要析构的目标 DeviceId 不同时，
// 内部正确切换 Device 完成销毁并恢复原 Device
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsDestroy_CrossDevice_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr int32_t curThreadDevId = 0;   // 当前线程的 DeviceId
    constexpr int32_t otherDevId = 1;       // 要析构的 DeviceId

    // 在 otherDevId 上创建 CcuInstance
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(otherDevId));
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&otherDevId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(otherDevId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockCcuNetworkDeviceDefault(otherDevId);
    EXPECT_EQ(MockCcuResourcesDefault(otherDevId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    CcuResDesc resDesc{};
    resDesc.dieId = hcomm::CCU_MAX_IODIE_NUM;
    resDesc.insType = CcuInstanceType::CCU_MS;
    constexpr uint32_t descNum = 1;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(static_cast<void *>(&resDesc), descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // isDiffDevId=true 场景
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(curThreadDevId));
    MOCKER(hrtSetDevice).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    // 销毁 CcuInstance
    ccuRet = HcommCcuInsDestroy(insHandle, otherDevId);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}
