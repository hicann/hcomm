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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "aicputs_hccs_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "endpoint.h"
#include "adapter_rts.h"
#include "hccl_net_dev.h"
#include "hccl_network.h"
#include "hccl_socket.h"
#include "network_manager_pub.h"
#include "reged_mem_mgr.h"
#include "mem_name_repository_pub.h"
#include "global_net_dev_manager.h"

using namespace hcomm;
using namespace hccl;

namespace {
static s32 deviceCurLogicId_ = 0;
static s32 deviceCurPhyId_ = 0;

void StubSetDevice(s32 deviceLogicId)
{
    deviceCurLogicId_ = deviceLogicId;
    deviceCurPhyId_ = deviceLogicId;
}

HcclResult StubHrtGetDevice(s32* deviceLogicId)
{
    if (deviceLogicId != nullptr) {
        *deviceLogicId = deviceCurLogicId_;
    }
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDeviceRefresh(s32* deviceLogicId)
{
    if (deviceLogicId != nullptr) {
        *deviceLogicId = deviceCurLogicId_;
    }
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDevicePhyIdByIndex(u32 deviceLogicId, u32& devicePhyId, bool isRefresh)
{
    devicePhyId = deviceLogicId;
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDeviceIndexByPhyId(u32 devicePhyId, u32& deviceLogicId)
{
    deviceLogicId = devicePhyId;
    return HCCL_SUCCESS;
}

HcclResult StubHcclSocketAcceptForEp(
    hccl::HcclSocket* /*self*/, const std::string& /*tag*/, std::shared_ptr<hccl::HcclSocket>& socket,
    u32 /*acceptTimeOut*/)
{
    socket = std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(nullptr), 16666);
    return HCCL_SUCCESS;
}

HcclResult StubGetDeviceVnicIP(u32 devicePhyId, u32 superDeviceId, hccl::HcclIpAddress& vnicIP)
{
    std::string ip = "127.0.0." + std::to_string(devicePhyId + 1);
    (void)vnicIP.SetReadableAddress(ip);
    return HCCL_SUCCESS;
}

HcclResult StubHcclNetOpenDev(
    HcclNetDevCtx* netDevCtx, NicType nicType, s32 devicePhyId, s32 deviceLogicId, hccl::HcclIpAddress localIp,
    hccl::HcclIpAddress backupIp)
{
    static hccl::NetDevContext kNetDevCtx[MAX_MODULE_DEVICE_NUM];
    static bool initialized[MAX_MODULE_DEVICE_NUM] = {false};
    if (!initialized[devicePhyId]) {
        hccl::HcclIpAddress localIp;
        std::string ip = "127.0.0." + std::to_string(devicePhyId + 1);
        (void)localIp.SetReadableAddress(ip);
        kNetDevCtx[devicePhyId].Init(NicType::VNIC_TYPE, 0, 0, localIp);
        initialized[devicePhyId] = true;
    }
    *netDevCtx = reinterpret_cast<HcclNetDev>(&kNetDevCtx[devicePhyId]);
    return HCCL_SUCCESS;
}

void StubHcclNetCloseDev(HcclNetDevCtx netDevCtx) {}

class AiCpuTsHccsEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AiCpuTsHccsEndpointTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AiCpuTsHccsEndpointTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(hrtEnableP2P).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtDisableP2P).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(StubHrtGetDevice));
        MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(StubHrtGetDeviceRefresh));
        MOCKER(hrtGetDevicePhyIdByIndex)
            .stubs()
            .with(mockcpp::any(), outBound(0U))
            .will(invoke(StubHrtGetDevicePhyIdByIndex));
        MOCKER(hrtGetDeviceIndexByPhyId)
            .stubs()
            .with(mockcpp::any(), outBound(0U))
            .will(invoke(StubHrtGetDeviceIndexByPhyId));
        MOCKER(HcclNetOpenDev).stubs().will(invoke(StubHcclNetOpenDev));
        MOCKER(HcclNetCloseDev).stubs().will(invoke(StubHcclNetCloseDev));
        MOCKER(&hccl::HcclSocket::Accept).stubs().will(invoke(StubHcclSocketAcceptForEp));

        MOCKER_CPP(&GlobalNetDevMgr::GetDeviceVnicIP).stubs().will(invoke(StubGetDeviceVnicIP));
        MOCKER_CPP(&MemNameRepository::SetIpcMem, HcclResult(MemNameRepository::*)(void*, u64, u8*, u32))
            .stubs()
            .will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&MemNameRepository::FindIpcMem).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&MemNameRepository::OpenIpcMem).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&MemNameRepository::CloseIpcMem).stubs().will(returnValue(HCCL_SUCCESS));

        std::cout << "A Test case in AiCpuTsHccsEndpointTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AiCpuTsHccsEndpointTest TearDown" << std::endl;
    }

    void SetEndpointDesc(uint32_t id, uint32_t devPhyId, EndpointDesc& endpointDesc)
    {
        endpointDesc.protocol = COMM_PROTOCOL_HCCS;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_ID;
        endpointDesc.commAddr.id = id;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        endpointDesc.loc.device.devPhyId = devPhyId;
        endpointDesc.loc.device.superDevId = static_cast<uint32_t>(-1);
        endpointDesc.loc.device.serverIdx = 0;
        endpointDesc.loc.device.superPodIdx = 0;
    }

    HcommResult CreateHccsEndpoint(uint32_t id, uint32_t devPhyId, EndpointDesc& endpointDesc, void** endpointHandle)
    {
        SetEndpointDesc(id, devPhyId, endpointDesc);
        return HcommEndpointCreate(&endpointDesc, endpointHandle);
    }

    CommMem CreateCommMem(void* addr, size_t size, CommMemType type)
    {
        CommMem mem;
        mem.type = type;
        mem.size = size;
        mem.addr = addr;
        return mem;
    }
};

// 内存注册和解注册成功
TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Register_Memory_NORMAL_Expect_Return_SUCCESS)
{
    void* endpointHandle{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommEndpointDestroy(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 解注册空MemHandle
TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Unregister_Without_Register_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* invalidMemHandle = nullptr;
    ret = HcommMemUnreg(endpointHandle, invalidMemHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcommEndpointDestroy(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 一次reg多次unreg
TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Double_Unregister_After_Register_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    // 正常注册
    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 第一次注销，应该成功
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 第二次注销同一个memHandle，应该失败
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    ret = HcommEndpointDestroy(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 注册空指针内存
TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Register_Null_Memory_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem(nullptr, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcommEndpointDestroy(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 注册大小为0的内存
TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Register_Zero_Size_Memory_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 0, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcommEndpointDestroy(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 注册无效内存类型
TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Register_Invalid_MemType_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_INVALID);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcommEndpointDestroy(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 注销空指针
TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Unregister_Null_Handle_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 使用nullptr注销
    ret = HcommMemUnreg(endpointHandle, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcommEndpointDestroy(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Unregister_Wrong_Handle_Expect_Return_Error)
{
    void* endpointHandle1{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* endpointHandle2{nullptr};
    EndpointDesc endpointDesc2;
    StubSetDevice(1);
    ret = CreateHccsEndpoint(1, 1, endpointDesc2, &endpointHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem1 = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    CommMem mem2 = CreateCommMem((void*)0x20, 20, COMM_MEM_TYPE_DEVICE);
    void *memHandle1, *memHandle2;

    // 在第一个endpoint上注册内存
    StubSetDevice(0);
    ret = HcommMemReg(endpointHandle1, "memTag1", &mem1, &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 在第二个endpoint上注册内存
    StubSetDevice(1);
    ret = HcommMemReg(endpointHandle2, "memTag2", &mem2, &memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 使用第二个endpoint的memHandle2去第一个endpoint注销失败
    ret = HcommMemUnreg(endpointHandle1, memHandle2);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    // 使用第一个endpoint的memHandle1去第二个endpoint注销失败
    ret = HcommMemUnreg(endpointHandle2, memHandle1);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    // 正确注销：各自在自己的endpoint上注销自己的内存
    ret = HcommMemUnreg(endpointHandle1, memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle2, memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommEndpointDestroy(endpointHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommEndpointDestroy(endpointHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 重复内存注册重复销毁
TEST_F(AiCpuTsHccsEndpointTest, Ut_When_Double_Register_Unregister_Memory_Expect_Return_SUCCESS)
{
    void* endpointHandle{nullptr};
    EndpointDesc endpointDesc;

    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle1 = nullptr;
    void* memHandle2 = nullptr;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(memHandle1, nullptr);

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(memHandle2, nullptr);

    ret = HcommMemUnreg(endpointHandle, memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommMemUnreg(endpointHandle, memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommEndpointDestroy(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AiCpuTsHccsEndpointTest, Ut_When_export_import_Expect_Return_SUCCESS)
{
    void* endpointHandle1{nullptr};
    EndpointDesc endpointDesc;
    StubSetDevice(0);
    HcommResult ret = CreateHccsEndpoint(0, 0, endpointDesc, &endpointHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* endpointHandle2{nullptr};
    EndpointDesc endpointDesc2;
    StubSetDevice(1);
    ret = CreateHccsEndpoint(1, 1, endpointDesc2, &endpointHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem1 = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    CommMem mem2 = CreateCommMem((void*)0x20, 20, COMM_MEM_TYPE_DEVICE);
    void *memHandle1, *memHandle2;

    // 在第一个endpoint上注册内存
    ret = HcommMemReg(endpointHandle1, "memTag1", &mem1, &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 在第二个endpoint上注册内存
    ret = HcommMemReg(endpointHandle2, "memTag2", &mem2, &memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    CommMem outMem;

    ret = HcommMemExport(endpointHandle1, memHandle1, &memDesc, &memDescLen);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommMemImport(endpointHandle2, memDesc, memDescLen, &outMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommMemUnimport(endpointHandle2, memDesc, memDescLen);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 正确注销：各自在自己的endpoint上注销自己的内存
    ret = HcommMemUnreg(endpointHandle1, memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle2, memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommEndpointDestroy(endpointHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommEndpointDestroy(endpointHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

} // namespace
