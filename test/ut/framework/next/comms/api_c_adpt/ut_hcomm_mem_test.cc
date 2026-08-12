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
#include "endpoint.h"
#include "endpoint_map.h"
#include "roce_mem.h"
#include "hccp.h"

namespace {

// ENABLE_EXPERIMENTAL 路径下 HcommMemReg / HcommMemUnreg / ... 通过 GetNicOps()->xxx()
// 调用，不走虚函数。这里把 ctx（即 this）转回 Endpoint 指针再调虚函数，
// 从而正确委托到 FakeEndpoint 对应的 override 方法。
static int32_t FakeNicRegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle)
{
    auto* endpoint = static_cast<hcomm::Endpoint*>(ctx);
    return static_cast<int32_t>(endpoint->RegisterMemory(*mem, tag, handle));
}

static int32_t FakeNicUnregisterMemory(void* ctx, void* handle)
{
    auto* endpoint = static_cast<hcomm::Endpoint*>(ctx);
    return static_cast<int32_t>(endpoint->UnregisterMemory(handle));
}
static int32_t FakeNicMemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen)
{
    auto* endpoint = static_cast<hcomm::Endpoint*>(ctx);
    return static_cast<int32_t>(endpoint->MemoryExport(handle, desc, descLen));
}

static int32_t FakeNicMemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem)
{
    auto* endpoint = static_cast<hcomm::Endpoint*>(ctx);
    return static_cast<int32_t>(endpoint->MemoryImport(desc, descLen, outMem));
}

static int32_t FakeNicMemoryUnimport(void* ctx, const void* desc, uint32_t descLen)
{
    auto* endpoint = static_cast<hcomm::Endpoint*>(ctx);
    return static_cast<int32_t>(endpoint->MemoryUnimport(desc, descLen));
}

static HcommNicEndpointOps g_fakeNicEndpointOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    nullptr,                 // init
    nullptr,                 // destroy
    FakeNicRegisterMemory,   // registerMemory
    FakeNicUnregisterMemory, // unregisterMemory
    FakeNicMemoryExport,     // memoryExport
    FakeNicMemoryImport,     // memoryImport
    FakeNicMemoryUnimport,   // memoryUnimport
    nullptr,                 // getListenPort
};

// FakeEndpoint: 持有外部传入的 RegedMemMgr, Register/Unregister 委托给它。
// 用于跨 EP 共享 / 不共享 RegedMemMgr 的 C API 层测试。
class FakeEndpoint : public hcomm::Endpoint {
public:
    FakeEndpoint(const EndpointDesc& desc, std::shared_ptr<hcomm::RegedMemMgr> mgr)
        : hcomm::Endpoint(desc),
          mgr_(std::move(mgr))
    {
        // ENABLE_EXPERIMENTAL 路径通过 GetNicOps()->registerMemory(nicCtx_, ...) 调用，
        // 此处将 nicCtx_ 设为 this，nicOps_ 指向委托回虚函数的 ops，保证 mock 生效。
        SetNicEndpointCtx(&g_fakeNicEndpointOps, this);
    }

    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult ServerSocketListen(const uint32_t port) override { return HCCL_SUCCESS; }

    HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) override
    {
        return mgr_->RegisterMemory(mem, memTag, memHandle);
    }
    HcclResult UnregisterMemory(void* memHandle) override { return mgr_->UnregisterMemory(memHandle); }

    HcclResult MemoryExport(void*, void**, uint32_t*) override { return HCCL_E_NOT_SUPPORT; }
    HcclResult MemoryImport(const void*, uint32_t, HcommMem*) override { return HCCL_E_NOT_SUPPORT; }
    HcclResult MemoryUnimport(const void*, uint32_t) override { return HCCL_E_NOT_SUPPORT; }
    HcclResult GetAllMemHandles(void**, uint32_t*) override { return HCCL_E_NOT_SUPPORT; }

private:
    std::shared_ptr<hcomm::RegedMemMgr> mgr_;
};

EndpointDesc MakeHostRoceDesc()
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    return desc;
}

EndpointHandle MakeTestHandle()
{
    static uintptr_t counter = 0x10000;
    return reinterpret_cast<EndpointHandle>(counter++);
}

void RegisterRaMrMockSuccess(MrHandle fakeMrHandle)
{
    MOCKER(RaRegisterMr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&fakeMrHandle, sizeof(fakeMrHandle)))
        .will(returnValue(0));
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
}

std::shared_ptr<hcomm::RoceRegedMemMgr> MakeSharedMgr(RdmaHandle fakeRdmaHandle)
{
    auto mgr = std::make_shared<hcomm::RoceRegedMemMgr>();
    mgr->rdmaHandle_ = fakeRdmaHandle;
    return mgr;
}

} // namespace
#include "hcomm_c_adpt.h"

class TestHcommMem : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override
    {
        if (validEpHandle_ != nullptr) {
            (void)HcommEndpointDestroy(validEpHandle_);
            validEpHandle_ = nullptr;
        }
        for (auto h : injectedHandles_) {
            hcomm::HcommEndpointMap map;
            map.RemoveEndpoint(h);
        }
        injectedHandles_.clear();
        TestHcommCAdptBase::TearDown();
    }

    EndpointHandle InjectEndpoint(std::shared_ptr<hcomm::RegedMemMgr> mgr)
    {
        EndpointHandle h = MakeTestHandle();
        auto ep = std::make_unique<FakeEndpoint>(MakeHostRoceDesc(), std::move(mgr));
        hcomm::HcommEndpointMap map;
        map.AddEndpoint(h, std::move(ep));
        injectedHandles_.push_back(h);
        return h;
    }

private:
    std::vector<EndpointHandle> injectedHandles_;

    // 为非 endpoint 校验类用例构造一个真实 endpoint，使其能进入新的 ops 分发路径
    void CreateValidEndpoint()
    {
        if (validEpHandle_ != nullptr)
            return;
        EndpointDesc desc{};
        desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        desc.protocol = COMM_PROTOCOL_ROCE;
        (void)HcommEndpointCreate(&desc, &validEpHandle_);
    }

    EndpointHandle validEpHandle_{nullptr};
};

TEST_F(TestHcommMem, Ut_TestHcommMemReg_When_InvalidHandle_Return_HCCL_E_NOT_FOUND)
{
    CommMem mem;
    mem.addr = malloc(1024);
    mem.size = 1024;
    mem.type = COMM_MEM_TYPE_HOST;
    void* memHandle = nullptr;

    EndpointHandle invalidHandle = reinterpret_cast<EndpointHandle>(0x0FFFFFFFFFFFFFFF);
    HcommResult ret = HcommMemReg(invalidHandle, "test_mem", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    free(mem.addr);
}

TEST_F(TestHcommMem, Ut_TestHcommMemReg_When_MemHandleNullptr_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    CommMem mem;
    mem.addr = malloc(1024);
    mem.size = 1024;
    mem.type = COMM_MEM_TYPE_HOST;

    HcommResult ret = HcommMemReg(validEpHandle_, "test_mem", &mem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    free(mem.addr);
}

TEST_F(TestHcommMem, Ut_TestHcommMemUnreg_When_InvalidHandle_Return_HCCL_E_PTR)
{
    EndpointHandle invalidHandle = reinterpret_cast<EndpointHandle>(0xFFFFFFFFFFFFFFFF);
    HcommResult ret = HcommMemUnreg(invalidHandle, nullptr);
    // ENABLE_EXPERIMENTAL 路径先校验 endpoint，无效句柄返回 HCCL_E_NOT_FOUND
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcommMem, Ut_TestHcommMemExport_When_InvalidMemHandle_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    HcommResult ret = HcommMemExport(validEpHandle_, nullptr, &memDesc, &memDescLen);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemExport_When_OutputNullptr_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    void* dummyHandle = this;
    HcommResult ret = HcommMemExport(validEpHandle_, dummyHandle, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemImport_When_InvalidDesc_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    HcommMem outMem;
    HcommResult ret = HcommMemImport(validEpHandle_, nullptr, 0, &outMem);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemUnimport_When_InvalidParams_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    HcommResult ret = HcommMemUnimport(validEpHandle_, nullptr, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemImport_When_DescLenZero_Return_HCCL_E_PARA)
{
    CreateValidEndpoint();
    HcommMem outMem;
    char desc[8] = {0};
    HcommResult ret = HcommMemImport(validEpHandle_, desc, 0, &outMem);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcommMem, Ut_TestHcommMemGetAllMemHandles_When_Nullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommMemGetAllMemHandles(nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 用例1: 单 EP 同 buffer 多次注册, 第二次走 alias 复用
TEST_F(TestHcommMem, MemReg_When_SingleEP_SameBufferTwice_Expect_BothSuccessAndHandlesDiffer)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle epHandle = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x1000);
    mem.size = 4096;

    HcommMemHandle h1 = nullptr;
    HcommResult ret1 = HcommMemReg(epHandle, "tag1", &mem, &h1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_NE(h1, nullptr);

    HcommMemHandle h2 = nullptr;
    HcommResult ret2 = HcommMemReg(epHandle, "tag2", &mem, &h2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_NE(h2, nullptr);

    EXPECT_NE(h1, h2);
}

// 用例2: 跨 EP 不共享 (原架构), 同 buffer 各自注册, 两 EP 都成功 (stub 都返 0, 冗余但不验证次数)
TEST_F(TestHcommMem, MemReg_When_CrossEP_NotShared_Expect_BothSuccess)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr1 = MakeSharedMgr(fakeRdmaHandle);
    auto mgr2 = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle ep1 = InjectEndpoint(mgr1);
    EndpointHandle ep2 = InjectEndpoint(mgr2);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x5000);
    mem.size = 2048;

    HcommMemHandle h1 = nullptr;
    HcommResult ret1 = HcommMemReg(ep1, "t1", &mem, &h1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_NE(h1, nullptr);

    HcommMemHandle h2 = nullptr;
    HcommResult ret2 = HcommMemReg(ep2, "t2", &mem, &h2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_NE(h2, nullptr);

    EXPECT_NE(h1, h2);
}

// 用例3: 跨 EP 共享 (SPEC 改造后), 同 buffer 各注册, 第二次走 alias 复用
TEST_F(TestHcommMem, MemReg_When_CrossEP_Shared_Expect_BothSuccessAndHandlesDiffer)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle ep1 = InjectEndpoint(mgr);
    EndpointHandle ep2 = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x5000);
    mem.size = 2048;

    HcommMemHandle h1 = nullptr;
    HcommResult ret1 = HcommMemReg(ep1, "t1", &mem, &h1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_NE(h1, nullptr);

    HcommMemHandle h2 = nullptr;
    HcommResult ret2 = HcommMemReg(ep2, "t2", &mem, &h2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_NE(h2, nullptr);

    EXPECT_NE(h1, h2);
}

// 用例4: 跨 EP 共享后各注销
TEST_F(TestHcommMem, MemUnreg_When_CrossEP_Shared_Expect_BothSuccess)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle ep1 = InjectEndpoint(mgr);
    EndpointHandle ep2 = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x5000);
    mem.size = 2048;

    HcommMemHandle h1 = nullptr;
    ASSERT_EQ(HcommMemReg(ep1, "t1", &mem, &h1), HCCL_SUCCESS);
    HcommMemHandle h2 = nullptr;
    ASSERT_EQ(HcommMemReg(ep2, "t2", &mem, &h2), HCCL_SUCCESS);

    HcommResult ret1 = HcommMemUnreg(ep1, h1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    HcommResult ret2 = HcommMemUnreg(ep2, h2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
}

// 用例5: 跨 EP 共享父子 buffer (子集走 alias)
TEST_F(TestHcommMem, MemReg_When_CrossEP_Shared_ParentChild_Expect_BothSuccess)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle ep1 = InjectEndpoint(mgr);
    EndpointHandle ep2 = InjectEndpoint(mgr);

    CommMem memParent{};
    memParent.type = COMM_MEM_TYPE_HOST;
    memParent.addr = reinterpret_cast<void*>(0x1000);
    memParent.size = 4096;

    CommMem memChild{};
    memChild.type = COMM_MEM_TYPE_HOST;
    memChild.addr = reinterpret_cast<void*>(0x1000);
    memChild.size = 1024;

    HcommMemHandle hParent = nullptr;
    HcommResult retP = HcommMemReg(ep1, "parent", &memParent, &hParent);
    EXPECT_EQ(retP, HCCL_SUCCESS);
    EXPECT_NE(hParent, nullptr);

    HcommMemHandle hChild = nullptr;
    HcommResult retC = HcommMemReg(ep2, "child", &memChild, &hChild);
    EXPECT_EQ(retC, HCCL_SUCCESS);
    EXPECT_NE(hChild, nullptr);

    EXPECT_NE(hParent, hChild);
}
