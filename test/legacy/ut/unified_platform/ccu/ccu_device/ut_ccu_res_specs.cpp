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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <chrono>
#define private public
#define protected public
#include "ccu_res_specs.h"
#include "hccl_common_v2.h"

#undef private
#undef protected

using namespace Hccl;

class CcuResSpecsTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuResSpecsTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuResSpecsTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuResSpecsTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuResSpecsTest TearDown" << std::endl;
    }
};

HcclResult
AllocCkeStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    ckeInfos.clear();
    ResInfo ckeInfo(0, num);
    ckeInfos.push_back(ckeInfo);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
AllocXnStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    xnInfos.clear();
    ResInfo xnInfo(0, num);
    xnInfos.push_back(xnInfo);
    return HcclResult::HCCL_SUCCESS;
}

void MockDoOnce() // 提供给其他用例使用的ccu资源打桩，使用后会导致单例存在内存中，本用例不需要
{
    CustomChannelInfoIn inBuff{};
    inBuff.data.dataInfo.dataLen = 512;
    inBuff.data.dataInfo.dataArraySize = 64;

    MOCKER(HrtRaTlvRequestForCustomChannel)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), outBoundP(reinterpret_cast<void*>(&inBuff), sizeof(inBuff)),
            outBoundP(reinterpret_cast<void*>(&inBuff), sizeof(inBuff)));
    MOCKER(HrtGetDeviceCount).stubs().with().will(returnValue(8));
    MOCKER(HrtGetSocVer).stubs().with().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(0));
    DevType deviceType = DevType::DevType::DEV_TYPE_910A3;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(deviceType));

    std::unique_ptr<RdmaHandle> handle = std::make_unique<RdmaHandle>();
    RdmaHandle handlePtr = handle.get();
    MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any()).will(returnValue(handlePtr));
    // 避免RdmaHandle空指针

    vector<HrtDevEidInfo> eidInfoListStbu;
    HrtDevEidInfo eidInfo;
    eidInfo.name = "udma0";
    eidInfo.dieId = 0;
    eidInfo.funcId = 3;

    eidInfoListStbu.push_back(eidInfo);

    MOCKER(HrtRaGetDevEidInfoList).stubs().with(mockcpp::any()).will(returnValue(eidInfoListStbu));

    for (int i = 0; i < 2; i++) {
        CcuResSpecifications::GetInstance(0).dieEnableFlags[i] = true;
        CcuResSpecifications::GetInstance(0).resSpecs[i].loopEngineNum = 200;
        CcuResSpecifications::GetInstance(0).resSpecs[i].msNum = 1536;
        CcuResSpecifications::GetInstance(0).resSpecs[i].ckeNum = 1024;
        CcuResSpecifications::GetInstance(0).resSpecs[i].xnNum = 3072;
        CcuResSpecifications::GetInstance(0).resSpecs[i].gsaNum = 3072;
        CcuResSpecifications::GetInstance(0).resSpecs[i].instructionNum = 32768;
        CcuResSpecifications::GetInstance(0).resSpecs[i].missionNum = 16;
        CcuResSpecifications::GetInstance(0).resSpecs[i].channelNum = 128;
        CcuResSpecifications::GetInstance(0).resSpecs[i].jettyNum = 128;
        CcuResSpecifications::GetInstance(0).resSpecs[i].wqeBBNum = 4096;
        CcuResSpecifications::GetInstance(0).resSpecs[i].resourceAddr = 0xE7FFBF800000;
    }
}

void MockCcuDriverInterfaceReturnDieEnableStub(void* tlvHandle, u32 msgType, void* customIn, void* customOut)
{
    CustomChannelInfoOut* mockOutBuff = (CustomChannelInfoOut*)customOut;
    CcuDieInfo dieInfo;
    dieInfo.enableFlag = true;
    memcpy_s(mockOutBuff->data.dataInfo.dataArray, sizeof(CcuDieInfo), &dieInfo, sizeof(CcuDieInfo));
}

void MockCcuOneDieResource(
    CcuResSpecifications& ccuResSpecs, const int32_t devLogicId, const uint8_t dieId, const CcuVersion ccuVersion)
{
    MOCKER(HrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(MAX_MODULE_DEVICE_NUM));

    ccuResSpecs.Init();
    ccuResSpecs.ccuVersion = ccuVersion;
    ccuResSpecs.dieEnableFlags[dieId] = true;

    ccuResSpecs.resSpecs[dieId].loopEngineNum = 200;

    ccuResSpecs.resSpecs[dieId].msNum = 1536;
    ccuResSpecs.resSpecs[dieId].ckeNum = 1024;

    ccuResSpecs.resSpecs[dieId].xnNum = 3072;

    ccuResSpecs.resSpecs[dieId].gsaNum = 3072;

    ccuResSpecs.resSpecs[dieId].instructionNum = 32768;
    ccuResSpecs.resSpecs[dieId].missionNum = 16;

    ccuResSpecs.resSpecs[dieId].channelNum = 128;

    ccuResSpecs.resSpecs[dieId].jettyNum = 128;
    ccuResSpecs.resSpecs[dieId].wqeBBNum = 4096;

    ccuResSpecs.resSpecs[dieId].pfeNum = 10;

    ccuResSpecs.resSpecs[dieId].resourceAddr = 0xE7FFBF800000;

    ccuResSpecs.resSpecs[dieId].msId = 1;       // 非0即可
    ccuResSpecs.resSpecs[dieId].missionKey = 1; // 非0即可
}

void PrepareInitMocker()
{
    MOCKER(HrtRaTlvRequestForCustomChannel).stubs().will(invoke(MockCcuDriverInterfaceReturnDieEnableStub));
    MOCKER(HrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(MAX_MODULE_DEVICE_NUM));
}

void MockRtGetDeviceInfo(int64_t outVal)
{
    MOCKER(aclrtGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&outVal, sizeof(outVal)))
        .will(returnValue(ACL_SUCCESS));
}

TEST_F(CcuResSpecsTest, Ut_Init_When_CcuDriverOk_Expect_Return_Ok)
{
    PrepareInitMocker();

    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    ccuResSpecs.Init();
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        EXPECT_EQ(ccuResSpecs.dieEnableFlags[dieId], true);
    }
}

TEST_F(CcuResSpecsTest, Ut_Init_When_AX_Mainboard_Expect_IsAX_Equals_To_True)
{
    PrepareInitMocker();
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    int64_t outVal = 0b01101010;
    MockRtGetDeviceInfo(outVal);

    CcuResSpecifications ccuResSpecs;
    ccuResSpecs.Init();
    EXPECT_EQ(ccuResSpecs.isAX, true);
}

TEST_F(CcuResSpecsTest, Ut_Init_When_Not_AX_Mainboard_Expect_IsAX_Equals_To_False)
{
    PrepareInitMocker();
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    int64_t outVal = 0b00101010;
    MockRtGetDeviceInfo(outVal);

    CcuResSpecifications ccuResSpecs;
    ccuResSpecs.Init();
    EXPECT_EQ(ccuResSpecs.isAX, false);
}

TEST_F(CcuResSpecsTest, Ut_InitGetCcuVersion_When_CcuV1_Expect_Return_Ok)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);

    EXPECT_EQ(ccuResSpecs.GetCcuVersion(), CcuVersion::CCU_V1);
}

TEST_F(CcuResSpecsTest, Ut_GetDieEnableFlag_When_DieIsValid_Expect_Return_Ok)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);

    bool dieFlag = false;
    auto ret = ccuResSpecs.GetDieEnableFlag(dieId, dieFlag);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(dieFlag, true);
}

TEST_F(CcuResSpecsTest, Ut_GetDieEnableFlag_When_DieIsValidButDisable_Expect_Return_Ok)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);
    ccuResSpecs.dieEnableFlags[dieId] = false;

    bool dieFlag = false;
    auto ret = ccuResSpecs.GetDieEnableFlag(dieId, dieFlag);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(dieFlag, false);
}

TEST_F(CcuResSpecsTest, Ut_PublicFunc_When_DieIsInvalid_Expect_Return_ErrorPara)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);

    uint8_t invalidDieId = MAX_CCU_IODIE_NUM;
    uint64_t resourceAddr = 0;
    auto ret = ccuResSpecs.GetResourceAddr(invalidDieId, resourceAddr);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    EXPECT_EQ(resourceAddr, 0);
}

TEST_F(CcuResSpecsTest, Ut_PublicFunc_When_DieIsDisable_Expect_Return_ErrorPara)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);
    ccuResSpecs.dieEnableFlags[dieId] = false;

    uint8_t invalidDieId = MAX_CCU_IODIE_NUM;
    uint64_t resourceAddr = 0;
    auto ret = ccuResSpecs.GetResourceAddr(dieId, resourceAddr);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    EXPECT_EQ(resourceAddr, 0);
}

TEST_F(CcuResSpecsTest, Ut_GetResourceAddr_When_DieIsValid_Expect_Return_Ok)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);

    uint64_t resourceAddr = 0;
    auto ret = ccuResSpecs.GetResourceAddr(dieId, resourceAddr);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(resourceAddr, 0);
}

TEST_F(CcuResSpecsTest, Ut_GetResourceAddr_When_DieIsValidAndCcuV1_Expect_Return_Ok)
{
    // 该用例按正常ccu满规格设计
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);
    ccuResSpecs.ccuVersion = CcuVersion::CCU_V1;

    uint64_t xnBaseAddr = 0;
    auto ret = ccuResSpecs.GetXnBaseAddr(dieId, xnBaseAddr);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    const uint64_t correctOffset = CCU_V1_CCUM_OFFSET + 0x108000;
    EXPECT_EQ(xnBaseAddr, ccuResSpecs.resSpecs[dieId].resourceAddr + correctOffset);
}

TEST_F(CcuResSpecsTest, Ut_GetResSpec_When_InitOk_Expect_Return_Ok)
{
    using GetResSpecFunc = HcclResult (CcuResSpecifications::*)(const uint8_t, uint32_t&) const;
    constexpr GetResSpecFunc GET_RES_SPEC_FUNC_ARRAY[]
        = {&CcuResSpecifications::GetLoopEngineNum, &CcuResSpecifications::GetMsNum,
           &CcuResSpecifications::GetCkeNum,        &CcuResSpecifications::GetXnNum,
           &CcuResSpecifications::GetGsaNum,        &CcuResSpecifications::GetInstructionNum,
           &CcuResSpecifications::GetMissionNum,    &CcuResSpecifications::GetMsId,
           &CcuResSpecifications::GetMissionKey,    &CcuResSpecifications::GetChannelNum,
           &CcuResSpecifications::GetJettyNum,      &CcuResSpecifications::GetPfeReservedNum,
           &CcuResSpecifications::GetPfeNum,        &CcuResSpecifications::GetWqeBBNum};

    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);
    ccuResSpecs.ccuVersion = CcuVersion::CCU_V1;

    for (const auto& getFunc : GET_RES_SPEC_FUNC_ARRAY) {
        uint32_t capacity = 0;
        std::cout << "Test GetResSpecFunc: " << getFunc << std::endl;
        EXPECT_EQ((ccuResSpecs.*getFunc)(dieId, capacity), HcclResult::HCCL_SUCCESS);
        EXPECT_NE(capacity, 0);
    }
}
