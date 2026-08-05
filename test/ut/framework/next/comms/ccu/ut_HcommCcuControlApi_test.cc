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
#include <acl/acl.h>
#include "hccl_api_base_test.h"

#define private public
#define protected public

#include "log.h"
#include "adapter_rts.h"
#include "hcomm_c_adpt.h"
#include "op_base.h"

#include "ccu_device_pub.h"
#include "ccu_kernel_mgr.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_instance_mgr.h"
#include "ccu_device_res.h"
#include "ccu_res_batch_allocator.h"
#include "ccu_res_specs.h"
#include "ccu_comp.h"
#include "ccu_pfe_cfg_mgr.h"
#include "ccu_launch.h"
#include "ccu_res.h"

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
#include "ccu_kernel_impl/ccu_reduce_scatter_mesh1d_a6_demo.h"
#include "ccu_kernel_impl/ccu_groupcopy_demo.h"

#undef protected
#undef private

namespace hcomm {
HcclResult GetHcclVersionForCcuKernelMgr(int &hcclVersion);
}

namespace {
constexpr int32_t TEST_DEVICE_LOGIC_ID = 0;
constexpr int32_t OTHER_TEST_DEVICE_LOGIC_ID = 1;
constexpr int32_t HCCL_VERSION_USING_VALIDATE_AND_APPLY_DIE = 90100001;
int32_t g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
uint32_t g_deviceRefreshCalls = 0;
HcclResult g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;

HcclResult MockHrtGetDeviceRefresh(int32_t *deviceLogicId)
{
    ++g_deviceRefreshCalls;
    if (g_deviceRefreshResult != HcclResult::HCCL_SUCCESS) {
        return g_deviceRefreshResult;
    }
    if (deviceLogicId == nullptr) {
        return HcclResult::HCCL_E_PTR;
    }
    *deviceLogicId = g_runtimeDeviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MockHrtGetDevice(int32_t *deviceLogicId)
{
    if (deviceLogicId == nullptr) {
        return HcclResult::HCCL_E_PTR;
    }
    *deviceLogicId = g_runtimeDeviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MockHrtGetDevicePhyIdByIndex(uint32_t deviceLogicId, uint32_t &devicePhyId, bool)
{
    devicePhyId = deviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

void MockControlDeviceRefresh(int32_t deviceLogicId)
{
    g_runtimeDeviceLogicId = deviceLogicId;
    g_deviceRefreshCalls = 0;
    g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
    MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(MockHrtGetDeviceRefresh));
}
} // namespace

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
        MOCKER(hcomm::GetHcclVersionForCcuKernelMgr)
            .stubs()
            .with(outBound(HCCL_VERSION_USING_VALIDATE_AND_APPLY_DIE))
            .will(returnValue(HCCL_SUCCESS));
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

static void SetCcuResDescCcuMs(HcommCcuResDescHandle handle, hcomm::CcuVersion ccuVersion)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_LOOP, 8 * 8 * 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_CCU_BUF, 64 * 8 * 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    if (ccuVersion == hcomm::CcuVersion::CCU_V2) {
        ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_VARIABLE, 800);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_ADDRESS, 0);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    } else {
        ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_VARIABLE, 400);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_ADDRESS, 400);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    }
    ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_EVENT, 32 + 8 * 8 * 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_CCU_THREAD, 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

// 打桩当前线程 DeviceId 并初始化 ccu 资源环境，返回 fakeDeviceLogicId 供调用方使用
static int32_t MockCcuDeviceEnv(uint32_t fakeDevId, hcomm::CcuVersion fakeCcuVersion)
{
    MockControlDeviceRefresh(static_cast<int32_t>(fakeDevId));
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(static_cast<int32_t>(fakeDevId)));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs()
        .with(outBoundP(&fakeDeviceLogicId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    return fakeDeviceLogicId;
}

// 创建两个 die（dieId 0/1）的资源描述符并按 CcuMs 默认值填充
static void CreateCcuResDescsPair(
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM], hcomm::CcuVersion ccuVersion)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    ccuRet = HcommCcuInsResDescCreate(0, &resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescCreate(1, &resDescs[1]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    SetCcuResDescCcuMs(resDescs[0], ccuVersion);
    SetCcuResDescCcuMs(resDescs[1], ccuVersion);
}

// 销毁资源描述符数组
static void DestroyCcuResDescs(HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM])
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    ccuRet = HcommCcuInsResDescDestroy(resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescDestroy(resDescs[1]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}


#define CCU_FUNC_KERNEL_TEST(testName, demoFunc, expectRegisterSuccess)                                     \
TEST_F(HcommCcuControlApiTest, testName)                                                                    \
{                                                                                                           \
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;                                                           \
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;                                               \
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;                                 \
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);                                                      \
                                                                                                            \
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};                                      \
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);                                                        \
    constexpr uint32_t descNum = 2;                                                                         \
    CcuInsHandle insHandle{0};                                                                              \
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);                                              \
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
    ccuRet = HcommCcuInsDestroy(insHandle);                                                                 \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
    DestroyCcuResDescs(resDescs);                                                                           \
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
CCU_FUNC_KERNEL_TEST(Ut_A5MixedLoopCount_Expect_Success, CcuA5MixedLoopCountDemoKernel, true)
CCU_FUNC_KERNEL_TEST(Ut_LoopCfgDemo_Expect_Success, CcuLoopCfgDemoKernel, true)

TEST_F(HcommCcuControlApiTest, Ut_LoopObjectApi_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopV2GroupApi_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void *kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void *>(CcuV2LoopGroupDemoKernel);
    char *kernelFuncName = "CcuV2LoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_A6MixedLoopCount_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void *kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void *>(CcuA6MixedLoopCountDemoKernel);
    char *kernelFuncName = "CcuA6MixedLoopCountDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopV2GroupApi_When_V1_Expect_Fail)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void *kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void *>(CcuV2LoopGroupDemoKernel);
    char *kernelFuncName = "CcuV2LoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopCompatGroupApi_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void *kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void *>(CcuV2CompatLoopGroupDemoKernel);
    char *kernelFuncName = "CcuV2CompatLoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopConfigGroupApi_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void *kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void *>(CcuV2ConfigLoopGroupDemoKernel);
    char *kernelFuncName = "CcuV2ConfigLoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopMixedGroupApi_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void *kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void *>(CcuV2MixedLoopGroupDemoKernel);
    char *kernelFuncName = "CcuV2MixedLoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// 版本化 cfg 全路径(Loop(CcuLoopCfg)+LoopGroup(CcuLoopGroupCfg)) demo,V2 下建组成功
TEST_F(HcommCcuControlApiTest, Ut_LoopGroupCfgApi_When_V2_Expect_Success)
{
    // 整体打桩，处理ccu资源
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void *kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void *>(CcuV2LoopGroupCfgDemoKernel);
    char *kernelFuncName = "CcuV2LoopGroupCfgDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// 版本化 cfg C ABI 的 strict 校验:非法版本头(magic 未初始化)被拒绝,注册失败。
CCU_FUNC_KERNEL_TEST(Ut_LoopGroupCreateCfg_BadHeader_Expect_Fail, CcuLoopGroupCreateCfgBadHeaderDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_LoopGroupAddLoopCfg_BadHeader_Expect_Fail, CcuLoopGroupAddLoopCfgBadHeaderDemoKernel, false)

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRegister_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    // 整体打桩，处理ccu资源
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest,
    Ut_HcommCcuKernelRegister_When_ChannelDieDiffersFromInputDie_Expect_ReturnIsCCU_E_PARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcIp = 167772383; // mock EID 对应 die0
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(fakeDevId, 1, srcIp, dstIp, commEngine);

    CcuVarAddKernelArg demoArg{};
    demoArg.numA = 1;
    demoArg.numB = 2;
    demoArg.channelHandle = handlePair.second;
    const void *kernelArgs[] = {static_cast<CcuKernelArg>(&demoArg)};

    ASSERT_EQ(HcommCcuKernelRegisterStart(insHandle), CcuResult::CCU_SUCCESS);
    constexpr uint32_t inputDieId = 1;
    CcuKernelHandle kernelHandle{0};
    EXPECT_EQ(HcommCcuKernelRegister(insHandle, inputDieId, "ccu_alloc_demo",
        reinterpret_cast<void *>(CcuAllocDemoKernel), kernelArgs, 1, &kernelHandle),
        CcuResult::CCU_E_PARA);
    EXPECT_EQ(kernelHandle, 0U);
    EXPECT_EQ(HcommCcuKernelRegisterEnd(insHandle), CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelAddrArithV2_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 4;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuAddrArithV2DemoKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.channelHandle = handlePair.second;
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_addr_arith_v2_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    auto fakeThreadHandle = MockThreadAllocWithStream(commEngine);
    std::vector<uint64_t> taskArgs{};
    void *fakeTaskArgs = static_cast<void *>(taskArgs.data());
    uint32_t fakeArgNum = taskArgs.size();
    EXPECT_EQ(HcommCcuKernelLaunch(fakeThreadHandle, kernelHandle,
        fakeTaskArgs, fakeArgNum), CcuResult::CCU_SUCCESS);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelDoWhile_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 3;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedIfOuterElse_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 4;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedIfInnerElse_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 5;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelDoWhileUnified_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 6;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_DoWhileLabelStackPopForWhile_When_Adjacent_Expect_ReturnLabel)
{
    hcomm::CcuKernel kernel;
    hcomm::CcuRep::CcuInsGeneratorV1 insGen;
    kernel.SetInsGenerater(&insGen);
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
    hcomm::CcuRep::CcuInsGeneratorV1 insGen;
    kernel.SetInsGenerater(&insGen);
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
    hcomm::CcuRep::CcuInsGeneratorV1 insGen;
    kernel.SetInsGenerater(&insGen);
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
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 7;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRemoteRead_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 8;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRemoteWrite_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 9;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelAlloc_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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

    // 验证 HcommCcuGetTaskArgsNum：CcuAllocDemoKernel 调用 LoadArg(0) 和 LoadArg(1)，max(argId)+1=2
    uint32_t taskArgsNum = 0xFFFFFFFF;
    EXPECT_EQ(HcommCcuGetTaskArgsNum(kernelHandle, &taskArgsNum), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(taskArgsNum, 2U);

    MockChannelDestory(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelReduceScatterMesh1d_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    // 整体打桩，处理ccu资源
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 11;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelReduceScatterMesh1d_When_V2_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 15;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto &handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuReduceScatterMesh1dV2Kernel;
    ReduceScatterKernelArgV2 demoArg{};
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

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char *kernelFuncName = "ccu_reduce_scatter_mesh1d_a6_demo";
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedInIfIf_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 12;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelAllGatherMesh1dMem2Mem_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 13;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
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
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsDestroy 接口在线程 DeviceId 与要析构的目标 DeviceId 不同时，
// 内部正确切换 Device 完成销毁并恢复原 Device
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsDestroy_CrossDevice_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr int32_t curThreadDevId = 0;   // 当前线程的 DeviceId
    constexpr int32_t otherDevId = 1;       // 要析构的 DeviceId

    // 在 otherDevId 上创建 CcuInstance
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(static_cast<uint32_t>(otherDevId), fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MOCKER(hrtSetDevice).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    // 销毁 CcuInstance
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(otherDevId));
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

#define CCU_RELATIONAL_V2_KERNEL_TEST(testName, demoFunc, argType, argInit)                                 \
TEST_F(HcommCcuControlApiTest, testName)                                                                    \
{                                                                                                           \
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;                                                           \
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 13;                                              \
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;                                 \
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);                                                      \
                                                                                                            \
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};                                      \
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);                                                        \
    constexpr uint32_t descNum = 2;                                                                         \
    CcuInsHandle insHandle{0};                                                                              \
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);                                              \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
                                                                                                            \
    ccuRet = HcommCcuKernelRegisterStart(insHandle);                                                        \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
                                                                                                            \
    argType demoArg{};                                                                                      \
    argInit;                                                                                                \
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);                                           \
    const void *kernelArgs[] = {kernelArg};                                                                 \
    CcuKernelHandle kernelHandle{0};                                                                        \
    auto kernelFunc = reinterpret_cast<void *>(demoFunc);                                                   \
    constexpr uint32_t fakeDieId = 0;                                                                       \
    constexpr uint32_t kernelArgNum = 1;                                                                    \
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId, const_cast<char *>(#demoFunc),                    \
        kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);                                               \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);                                                          \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
                                                                                                            \
    ccuRet = HcommCcuInsDestroy(insHandle);                                                                 \
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                              \
    DestroyCcuResDescs(resDescs);                                                                           \
}

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelIfRelational_When_V2_Expect_ReturnCcuSUCCESS,
    CcuIfRelationalDemoKernel,
    CcuIfRelationalDemoKernelArg,
    (demoArg.value = 5, demoArg.bound = 10))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelIfElseRelational_When_V2_Expect_ReturnCcuSUCCESS,
    CcuIfElseRelationalDemoKernel,
    CcuIfElseRelationalDemoKernelArg,
    (demoArg.value = 7, demoArg.threshold = 5))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelWhileRelational_When_V2_Expect_ReturnCcuSUCCESS,
    CcuWhileRelationalDemoKernel,
    CcuWhileRelationalDemoKernelArg,
    (demoArg.loopCount = 5))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelDoWhileRelational_When_V2_Expect_ReturnCcuSUCCESS,
    CcuDoWhileRelationalDemoKernel,
    CcuDoWhileRelationalDemoKernelArg,
    (demoArg.loopCount = 5))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelVariableComputing_When_V2_Expect_ReturnCcuSUCCESS,
    CcuVariableComputingKernel,
    CcuVariableComputingKernelArg,
    (demoArg.varA = 1024, demoArg.varB = 2048))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelVarVarControlFlow_When_V2_Expect_ReturnCcuSUCCESS,
    CcuVarVarControlFlowDemoKernel,
    CcuVarVarControlFlowDemoKernelArg,
    (demoArg.valueA = 3, demoArg.valueB = 7, demoArg.loopCount = 4))

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelIfRelational_When_V1_Expect_RegisterEndFail)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 14;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuIfRelationalDemoKernelArg demoArg{};
    demoArg.value = 3;
    demoArg.bound = 5;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void *kernelArgs[] = {kernelArg};
    CcuKernelHandle kernelHandle{0};
    auto kernelFunc = reinterpret_cast<void *>(CcuIfRelationalDemoKernel);
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, fakeDieId,
        const_cast<char *>("CcuIfRelationalDemoKernel_V1"),
        kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // 翻译阶段：V1 generator 走 ValidateInsGeneratorForJump 抛 CcuApiException → 失败
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 基于资源描述符数组创建实例，正常路径返回 CCU_SUCCESS
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_NE(insHandle, 0);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 内部实例创建或登记失败时返回 CCU_E_INTERNAL，且不产生实例句柄
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_ManagerInternalError_Expect_ReturnCcuEINTERNAL)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockControlDeviceRefresh(static_cast<int32_t>(fakeDevId));
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(static_cast<int32_t>(fakeDevId)));

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    MOCKER_CPP(&hcomm::CcuInstanceMgr::CreateByResDescs).stubs()
        .will(returnValue(CcuResult::CCU_E_INTERNAL));

    CcuInsHandle insHandle{0};
    EXPECT_EQ(HcommCcuInsCreate(resDescs, hcomm::CCU_MAX_IODIE_NUM, &insHandle),
        CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(insHandle, 0);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 入参 ccuInsHandle 为空指针，返回 CCU_E_PTR
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_NullInsHandle_Expect_ReturnCcuEPTR)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    ccuRet = HcommCcuInsCreate(resDescs, descNum, nullptr);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PTR);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 入参 resDescNum 为 0，返回 CCU_E_PARA
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_ResDescNumZero_Expect_ReturnCcuEPARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, 0, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 入参两个 resDesc 的 dieId 重复，返回 CCU_E_PARA
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_DieIdDuplicated_Expect_ReturnCcuEPARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // 两个 resDesc 都归属 dieId 0
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    ccuRet = HcommCcuInsResDescCreate(0, &resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescCreate(0, &resDescs[1]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    SetCcuResDescCcuMs(resDescs[0], fakeCcuVersion);
    SetCcuResDescCcuMs(resDescs[1], fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 内部 CcuInitFeature 返回 CCU_E_DRV_BUSY（ccu 驱动拉起失败），
// 期望接口返回 CCU_E_DRV_BUSY，且不产生实例句柄
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_DrvBusy_Expect_ReturnCcuEDRVBUSY)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // 打桩 CcuInitFeature 模拟 ccu 驱动拉起失败
    MOCKER(hcomm::CcuInitFeature).stubs().will(returnValue(CcuResult::CCU_E_DRV_BUSY));

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_DRV_BUSY);
    EXPECT_EQ(insHandle, 0);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 内部 CcuResPack::InitByResDescs 返回 CCU_E_UNAVAIL（资源不足），
// 期望接口返回 CCU_E_UNAVAIL，且不产生实例句柄
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_ResUnavail_Expect_ReturnCcuEUNAVAIL)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // 打桩 CcuResPack::InitByResDescs 模拟资源不足
    MOCKER_CPP(&hcomm::CcuResPack::InitByResDescs).stubs()
        .will(returnValue(CcuResult::CCU_E_UNAVAIL));

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_UNAVAIL);
    EXPECT_EQ(insHandle, 0);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreateDefault 使用当前 Device 上所有已使能 ioDie 的全部资源，返回 CCU_SUCCESS
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreateDefault_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // dieIds/dieNum 为保留参数，调用方按契约传入 nullptr/0
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreateDefault(nullptr, 0, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_NE(insHandle, 0);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

// HcommCcuInsCreateDefault 入参 ccuInsHandle 为空指针，返回 CCU_E_PTR
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreateDefault_When_NullInsHandle_Expect_ReturnCcuEPTR)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    ccuRet = HcommCcuInsCreateDefault(nullptr, 0, nullptr);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PTR);
}

// HcommCcuInsCreateDefault 内部实例创建或登记失败时返回 CCU_E_INTERNAL，且不产生实例句柄
TEST_F(HcommCcuControlApiTest,
    Ut_HcommCcuInsCreateDefault_When_ManagerInternalError_Expect_ReturnCcuEINTERNAL)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    MockControlDeviceRefresh(static_cast<int32_t>(fakeDevId));
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(static_cast<int32_t>(fakeDevId)));
    MOCKER_CPP(&hcomm::CcuInstanceMgr::CreateByAllRes).stubs()
        .will(returnValue(CcuResult::CCU_E_INTERNAL));

    CcuInsHandle insHandle{0};
    EXPECT_EQ(HcommCcuInsCreateDefault(nullptr, 0, &insHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(insHandle, 0);
}

// HcommCcuInsQueryResDesc 查询实例在指定 die 上的占用资源，正常路径返回 CCU_SUCCESS，
// 且查询到的占用数量应为非负值
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsQueryResDesc_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // 用 resDescs[0]（归属 dieId 0）作为查询载体，查询该实例在 dieId 0 上的占用资源
    ccuRet = HcommCcuInsQueryResDesc(insHandle, resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    uint32_t resNum = 0;
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_LOOP, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 8 * 8 * 2);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_CCU_BUF, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 64 * 8 * 2);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_VARIABLE, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 400);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_ADDRESS, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 400);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_EVENT, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 32 + 8 * 8 * 2);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_CCU_THREAD, &resNum);
    EXPECT_EQ(resNum, 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsQueryResDesc 入参 ccuInsHandle 为 0，返回 CCU_E_PARA
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsQueryResDesc_When_InvalidInsHandle_Expect_ReturnCcuEPARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    ccuRet = HcommCcuInsQueryResDesc(0, resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsQueryResDesc 入参 resDesc 为 0，返回 CCU_E_PARA
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsQueryResDesc_When_NullResDesc_Expect_ReturnCcuEPARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsQueryResDesc(insHandle, 0);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

class HcommCcuInstanceDeviceRefreshTest : public BaseInit {
public:
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        BaseInit::SetUp();
        ResetDeviceState(TEST_DEVICE_LOGIC_ID);
        ResetDeviceState(OTHER_TEST_DEVICE_LOGIC_ID);

        MOCKER(GetExternalInputHcclEnableEntryLog)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(true));
        MockControlDeviceRefresh(TEST_DEVICE_LOGIC_ID);
        int32_t seededDeviceLogicId = INVALID_INT;
        ASSERT_EQ(HcclDeviceRefresh(seededDeviceLogicId), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(seededDeviceLogicId, TEST_DEVICE_LOGIC_ID);
        ASSERT_EQ(HcclGetThreadDeviceId(), TEST_DEVICE_LOGIC_ID);
        g_deviceRefreshCalls = 0;

        MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(MockHrtGetDevice));
        MOCKER(hrtGetDevicePhyIdByIndex)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(MockHrtGetDevicePhyIdByIndex));
        constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
        MockCcuNetworkDeviceDefault(TEST_DEVICE_LOGIC_ID);
        ASSERT_EQ(MockCcuResourcesDefault(TEST_DEVICE_LOGIC_ID, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(InitMockCcuResourcesForDevice(
            OTHER_TEST_DEVICE_LOGIC_ID, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
        MockCcuChannelGetRes();
        MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
        ASSERT_EQ(hcomm::CcuInstanceMgr::GetInstance(TEST_DEVICE_LOGIC_ID).Init(), CcuResult::CCU_SUCCESS);
        ASSERT_EQ(hcomm::CcuInstanceMgr::GetInstance(
            OTHER_TEST_DEVICE_LOGIC_ID).Init(), CcuResult::CCU_SUCCESS);
    }

    void TearDown() override
    {
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
        int32_t restoredDeviceLogicId = INVALID_INT;
        EXPECT_EQ(HcclDeviceRefresh(restoredDeviceLogicId), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(restoredDeviceLogicId, TEST_DEVICE_LOGIC_ID);
        ResetDeviceState(TEST_DEVICE_LOGIC_ID);
        ResetDeviceState(OTHER_TEST_DEVICE_LOGIC_ID);
        BaseInit::TearDown();
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

protected:
    static void ResetDeviceState(int32_t deviceLogicId)
    {
        (void)hcomm::CcuInstanceMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuKernelMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuResBatchAllocator::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuComponent::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuPfeCfgMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuResSpecifications::GetInstance(deviceLogicId).Deinit();
    }

    static hcomm::CcuInstanceMgr &InsMgr(int32_t deviceLogicId)
    {
        return hcomm::CcuInstanceMgr::GetInstance(deviceLogicId);
    }

    static void CreateResDescsDirect(
        int32_t deviceLogicId, HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM])
    {
        auto &resDescMgr = InsMgr(deviceLogicId).GetResDescMgr();
        for (uint32_t dieId = 0; dieId < hcomm::CCU_MAX_IODIE_NUM; ++dieId) {
            ASSERT_EQ(resDescMgr.Create(dieId, resDescs[dieId]), CcuResult::CCU_SUCCESS);
            ASSERT_NE(resDescs[dieId], 0U);
            ASSERT_EQ(resDescMgr.SetResNum(
                resDescs[dieId], hcomm::ResType::LOOP, 8 * 8 * 2), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(
                resDescs[dieId], hcomm::ResType::MS, 64 * 8 * 2), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(
                resDescs[dieId], hcomm::ResType::XN, 400), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(
                resDescs[dieId], hcomm::ResType::GSA, 400), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(
                resDescs[dieId], hcomm::ResType::CKE, 32 + 8 * 8 * 2), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(
                resDescs[dieId], hcomm::ResType::MISSION, 2), CcuResult::CCU_SUCCESS);
        }
    }
};

// 验证实例创建、资源查询和销毁均使用刷新后的运行时 Device。
TEST_F(HcommCcuInstanceDeviceRefreshTest,
    Ut_HcommCcuInstanceApis_When_RuntimeDeviceSwitches_Expect_UseRefreshedDeviceForCreateQueryDestroy)
{
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateResDescsDirect(OTHER_TEST_DEVICE_LOGIC_ID, resDescs);
    g_runtimeDeviceLogicId = OTHER_TEST_DEVICE_LOGIC_ID;

    CcuInsHandle insHandle = 0;
    EXPECT_EQ(HcommCcuInsCreate(
        resDescs, hcomm::CCU_MAX_IODIE_NUM, &insHandle), CcuResult::CCU_SUCCESS);
    ASSERT_NE(insHandle, 0U);
    EXPECT_EQ(InsMgr(TEST_DEVICE_LOGIC_ID).Get(insHandle), nullptr);
    EXPECT_NE(InsMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(insHandle), nullptr);
    EXPECT_EQ(HcommCcuInsQueryResDesc(insHandle, resDescs[0]), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(HcommCcuInsDestroy(insHandle), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(InsMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(insHandle), nullptr);
    EXPECT_EQ(g_deviceRefreshCalls, 3U);
}

// 验证 Legacy 和 Default 创建入口在运行时 Device 切换后均先刷新并路由到新 Device。
TEST_F(HcommCcuInstanceDeviceRefreshTest,
    Ut_HcommCcuDefaultAndLegacyCreate_When_RuntimeDeviceSwitches_Expect_UseRefreshedDevice)
{
    g_runtimeDeviceLogicId = OTHER_TEST_DEVICE_LOGIC_ID;

    CcuInsHandle legacyHandle = 0;
    EXPECT_EQ(HcommCcuInsCreateLegacy(
        CcuInstanceType::CCU_SCHED, &legacyHandle), CcuResult::CCU_SUCCESS);
    ASSERT_NE(legacyHandle, 0U);
    EXPECT_EQ(InsMgr(TEST_DEVICE_LOGIC_ID).Get(legacyHandle), nullptr);
    EXPECT_NE(InsMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(legacyHandle), nullptr);
    EXPECT_EQ(g_deviceRefreshCalls, 1U);
    EXPECT_EQ(HcommCcuInsDestroy(legacyHandle), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(g_deviceRefreshCalls, 2U);

    CcuInsHandle defaultHandle = 0;
    // Mock 资源池无法满足 Default 接口的全量资源申请，因此预期返回 CCU_E_UNAVAIL；
    // 通过刷新次数和线程 DeviceId 验证接口已路由到切换后的 Device。
    EXPECT_EQ(HcommCcuInsCreateDefault(nullptr, 0, &defaultHandle), CcuResult::CCU_E_UNAVAIL);
    EXPECT_EQ(defaultHandle, 0U);
    EXPECT_EQ(HcclGetThreadDeviceId(), OTHER_TEST_DEVICE_LOGIC_ID);
    EXPECT_EQ(g_deviceRefreshCalls, 3U);
}

// 验证 Device 刷新失败时，所有实例生命周期入口立即返回错误且不改变管理器状态。
TEST_F(HcommCcuInstanceDeviceRefreshTest,
    Ut_HcommCcuInstanceApis_When_DeviceRefreshFails_Expect_ReturnErrorWithoutManagerMutation)
{
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateResDescsDirect(TEST_DEVICE_LOGIC_ID, resDescs);
    CcuInsHandle existingHandle = 0;
    ASSERT_EQ(HcommCcuInsCreateLegacy(
        CcuInstanceType::CCU_SCHED, &existingHandle), CcuResult::CCU_SUCCESS);
    ASSERT_NE(existingHandle, 0U);
    ASSERT_NE(InsMgr(TEST_DEVICE_LOGIC_ID).Get(existingHandle), nullptr);

    uint32_t originalLoopNum = 0;
    ASSERT_EQ(InsMgr(TEST_DEVICE_LOGIC_ID).GetResDescMgr().QueryResNum(
        resDescs[0], hcomm::ResType::LOOP, originalLoopNum), CcuResult::CCU_SUCCESS);
    g_deviceRefreshResult = HcclResult::HCCL_E_INTERNAL;
    g_deviceRefreshCalls = 0;

    CcuInsHandle createHandle = 0x1111;
    CcuInsHandle defaultHandle = 0x2222;
    CcuInsHandle legacyHandle = 0x3333;
    EXPECT_EQ(HcommCcuInsCreate(
        resDescs, hcomm::CCU_MAX_IODIE_NUM, &createHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(createHandle, 0x1111U);
    EXPECT_EQ(HcommCcuInsCreateDefault(nullptr, 0, &defaultHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(defaultHandle, 0x2222U);
    EXPECT_EQ(HcommCcuInsQueryResDesc(
        existingHandle, resDescs[0]), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(HcommCcuInsDestroy(existingHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(HcommCcuInsCreateLegacy(
        CcuInstanceType::CCU_SCHED, &legacyHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(legacyHandle, 0x3333U);

    uint32_t loopNum = 0;
    EXPECT_EQ(InsMgr(TEST_DEVICE_LOGIC_ID).GetResDescMgr().QueryResNum(
        resDescs[0], hcomm::ResType::LOOP, loopNum), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(loopNum, originalLoopNum);
    EXPECT_NE(InsMgr(TEST_DEVICE_LOGIC_ID).Get(existingHandle), nullptr);
    EXPECT_EQ(g_deviceRefreshCalls, 5U);
}

CCU_FUNC_KERNEL_TEST(Ut_HcommCcuKernelLocalCopy_When_AllFine_Expect_ReturnCcuSUCCESS, CcuLocalCopyKernel, true)

// ============================================================================================
// HcommCcuGetTaskArgsNum 单元测试
// 对应业务代码：src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc
// 该接口取已注册 kernel 注册期间所有 LoadArg 中 argId 的最大值加1，写入 *taskArgsNum。
// 调用链：HcommCcuGetTaskArgsNum -> CcuKernelMgr::GetInstance -> GetCcuKernelInfo -> CcuKernel::GetCcuKernelInfo
// 由于 #define private public 已启用，可直接操作单例 kernelMap_ 注入 fake kernel 精确测试
// ============================================================================================

// 功能用例：loadArgUsedSet_ 含多个 argId，验证返回最大值
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuGetTaskArgsNum_When_MultiArgIds_Expect_MaxArgId) {
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));

    // 通过单例注入 fake kernel，设置 loadArgUsedSet_ = {0, 3, 1}，max=3，+1=4
    constexpr CcuKernelHandle fakeHandle = 0xBEEF;
    auto &kernelMgr = hcomm::CcuKernelMgr::GetInstance(static_cast<int32_t>(fakeDevId));
    auto fakeKernel = std::make_unique<hcomm::CcuKernel>();
    fakeKernel->loadArgUsedSet_ = {0, 3, 1};
    kernelMgr.kernelMap_[fakeHandle] = std::move(fakeKernel);

    uint32_t taskArgsNum = 0xFFFFFFFF;
    CcuResult ret = HcommCcuGetTaskArgsNum(fakeHandle, &taskArgsNum);
    EXPECT_EQ(ret, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(taskArgsNum, 4U) << "应返回 loadArgUsedSet_ 中最大 argId + 1";

    // 清理 fake kernel，避免单例状态污染后续用例
    kernelMgr.kernelMap_.erase(fakeHandle);
}

// 异常用例：taskArgsNum 为空指针，应返回 CCU_E_PTR
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuGetTaskArgsNum_When_NullPtr_Expect_CcuEPtr) {
    constexpr CcuKernelHandle fakeHandle = 0xBEEF;
    CcuResult ret = HcommCcuGetTaskArgsNum(fakeHandle, nullptr);
    EXPECT_EQ(ret, CcuResult::CCU_E_PTR);
}

// 异常用例：kernelHandle 不存在，GetCcuKernelInfo 在锁内查找失败返回 CCU_E_NOT_FOUND
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuGetTaskArgsNum_When_InvalidHandle_Expect_CcuENotFound) {
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));

    // 不注册任何 kernel，使用不存在的 handle
    constexpr CcuKernelHandle invalidHandle = 0xDEAD;
    uint32_t taskArgsNum = 0xFFFFFFFF;
    CcuResult ret = HcommCcuGetTaskArgsNum(invalidHandle, &taskArgsNum);
    EXPECT_EQ(ret, CcuResult::CCU_E_NOT_FOUND);
    // 出参不应被修改（GetCcuKernelInfo 在锁内查找失败，未写入 info）
    EXPECT_EQ(taskArgsNum, 0xFFFFFFFFu);
}

// 安全/边界用例：未调用 LoadArg（loadArgUsedSet_ 为空），应返回 0 并覆盖脏出参
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuGetTaskArgsNum_When_NoLoadArg_Expect_Zero) {
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));

    // 注入 fake kernel，loadArgUsedSet_ 保持默认空集，GetCcuKernelInfo 填充 maxTaskArgsNum=0
    constexpr CcuKernelHandle fakeHandle = 0xBEEF;
    auto &kernelMgr = hcomm::CcuKernelMgr::GetInstance(static_cast<int32_t>(fakeDevId));
    auto fakeKernel = std::make_unique<hcomm::CcuKernel>();
    // loadArgUsedSet_ 默认为空，不 insert 任何 argId
    kernelMgr.kernelMap_[fakeHandle] = std::move(fakeKernel);

    // 出参初始为脏值 0xFFFFFFFF，验证被正确覆盖为 0
    uint32_t taskArgsNum = 0xFFFFFFFF;
    CcuResult ret = HcommCcuGetTaskArgsNum(fakeHandle, &taskArgsNum);
    EXPECT_EQ(ret, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(taskArgsNum, 0U) << "未调用 LoadArg 时 maxArgId 应为 0，脏出参应被覆盖";

    // 清理 fake kernel
    kernelMgr.kernelMap_.erase(fakeHandle);
}
