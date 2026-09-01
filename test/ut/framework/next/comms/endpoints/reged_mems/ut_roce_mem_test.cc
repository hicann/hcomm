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
#include "endpoint.h"
#include "endpoint_pair.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "invalid_params_exception.h"
#include "null_ptr_exception.h"

#define private public
#define protected public

#include "roce_reged_mem_mgr.h"

#undef protected
#undef private

using namespace hcomm;

class RoceRegedMemMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RoceRegedMemMgrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RoceRegedMemMgrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in RoceRegedMemMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RoceRegedMemMgrTest TearDown" << std::endl;
    }
};

template <typename Mgr, typename Key>
static uint64_t GetRef(Mgr& mgr, const Key& key)
{
    for (auto it = mgr.Begin(); it != mgr.End(); ++it) {
        if (it->first == key) {
            return it->second.ref;
        }
    }
    return 0;
}

TEST_F(RoceRegedMemMgrTest, Ut_GetParamsFromMemDesc_When_DescLenTooSmall_Expect_Return_Error)
{
    std::shared_ptr<RoceRegedMemMgr> roceRegedMemMgrPtr = std::make_shared<RoceRegedMemMgr>(nullptr);
    EndpointDesc endpointDesc;
    Hccl::ExchangeRdmaBufferDto dto;

    char buffer[10];
    uint32_t descLen = 10;

    HcclResult ret = roceRegedMemMgrPtr->GetParamsFromMemDesc(buffer, descLen, endpointDesc, dto);
    EXPECT_EQ(HCCL_E_INTERNAL, ret);
}

// 父+子集注册 → 先解注册子(alias) → 再解注册父
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_ParentChild_Expect_UnregisterChildFirst)
{
    RoceRegedMemMgr roceRegedMemMgr{nullptr};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = (void*)0x1000;
    mem0.size = 4096;
    std::string memTag0 = "parent";
    void* memHandle0 = nullptr;
    HcclResult ret = roceRegedMemMgr.RegisterMemory(&mem0, memTag0.c_str(), &memHandle0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto* parentBuf = static_cast<Hccl::LocalRdmaRmaBuffer*>(memHandle0);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_TRUE(roceRegedMemMgr.localRdmaRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 1u);

    // 子集注册（应创建alias buffer）
    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x1000;
    mem1.size = 1024;
    std::string memTag1 = "child";
    void* memHandle1 = nullptr;
    ret = roceRegedMemMgr.RegisterMemory(&mem1, memTag1.c_str(), &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(memHandle1, memHandle0);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 2u);

    // 解注册子（ref 2→1）
    ret = roceRegedMemMgr.UnregisterMemory(memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(roceRegedMemMgr.localRdmaRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 1u);

    // 解注册父（ref 1→0）
    ret = roceRegedMemMgr.UnregisterMemory(memHandle0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(roceRegedMemMgr.localRdmaRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 0u);
}

// 多次注册同一范围，各自解注册
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_MultipleSameRange_Expect_AllSuccess)
{
    RoceRegedMemMgr roceRegedMemMgr{nullptr};
    HcommMem mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = (void*)0x5000;
    mem.size = 2048;

    void *h1 = nullptr, *h2 = nullptr, *h3 = nullptr;
    EXPECT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t1", &h1), HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalRdmaRmaBuffer*>(h1);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 1u);

    EXPECT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t2", &h2), HCCL_SUCCESS);
    EXPECT_NE(h2, h1);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 2u);

    EXPECT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t3", &h3), HCCL_SUCCESS);
    EXPECT_NE(h3, h1);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 3u);

    // ref 3→2
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h3), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 2u);
    // ref 2→1
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 1u);
    // ref 1→0
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 0u);
}

TEST_F(RoceRegedMemMgrTest, Ut_MemoryExport_When_MemHandleUnregistered_Expect_NotFound)
{
    RoceRegedMemMgr roceRegedMemMgr{nullptr};
    EndpointDesc endpointDesc{};

    auto buf = std::make_shared<Hccl::Buffer>(0xD900, 0x100, HCCL_MEM_TYPE_DEVICE, "roce_export");
    auto localRdmaRmaBuffer = std::make_shared<Hccl::LocalRdmaRmaBuffer>(
        buf, reinterpret_cast<RdmaHandle>(0x1), 11U, 22U, reinterpret_cast<MrHandle>(0x2));
    void* memHandle = localRdmaRmaBuffer.get();

    roceRegedMemMgr.allRegisteredBuffers_.emplace_back(localRdmaRmaBuffer, false);
    roceRegedMemMgr.allRegisteredBuffers_.clear();

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(roceRegedMemMgr.MemoryExport(endpointDesc, memHandle, &memDesc, &memDescLen), HCCL_E_NOT_FOUND);
    EXPECT_EQ(memDesc, nullptr);
    EXPECT_EQ(memDescLen, 0U);
}

TEST_F(RoceRegedMemMgrTest, Ut_GetParamsFromMemDesc_When_DescLenEqualSize_Expect_Return_Success)
{
    std::shared_ptr<RoceRegedMemMgr> roceRegedMemMgrPtr = std::make_shared<RoceRegedMemMgr>(nullptr);
    EndpointDesc endpointDesc;
    Hccl::ExchangeRdmaBufferDto dto;

    char buffer[sizeof(EndpointDesc)];
    uint32_t descLen = sizeof(EndpointDesc);

    MOCKER_CPP_VIRTUAL(dto, &Hccl::ExchangeRdmaBufferDto::Deserialize).stubs();

    HcclResult ret = roceRegedMemMgrPtr->GetParamsFromMemDesc(buffer, descLen, endpointDesc, dto);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

// 父+子集注册 → 先解注册父(soft-delete) → 验证handlesRecords_和allBuffers_状态 → 再解注册子
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_UnregisterParentFirst_Expect_ParentSoftDeleted)
{
    RoceRegedMemMgr roceRegedMemMgr{nullptr};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = (void*)0x7000;
    mem0.size = 4096;
    void* parentHandle = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem0, "parent", &parentHandle), HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalRdmaRmaBuffer*>(parentHandle);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 1u);
    EXPECT_EQ(roceRegedMemMgr.handlesRecords_.size(), 1u);

    // 子集注册（alias）
    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x7000;
    mem1.size = 512;
    void* childHandle = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem1, "child", &childHandle), HCCL_SUCCESS);
    EXPECT_NE(childHandle, parentHandle);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 2u);
    EXPECT_EQ(roceRegedMemMgr.handlesRecords_.size(), 2u);
    EXPECT_EQ(roceRegedMemMgr.allRegisteredBuffers_.size(), 2u);

    // 先解注册父（ref 2→1，IsInTree=true，父标记为soft-deleted）
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(parentHandle), HCCL_SUCCESS);
    EXPECT_TRUE(roceRegedMemMgr.localRdmaRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 1u);

    // 验证：handlesRecords_中已移除父，但allBuffers_中父标记为soft-deleted
    EXPECT_EQ(roceRegedMemMgr.handlesRecords_.size(), 1u);
    EXPECT_EQ(roceRegedMemMgr.allRegisteredBuffers_.size(), 2u);

    auto itParent = std::find_if(
        roceRegedMemMgr.allRegisteredBuffers_.begin(), roceRegedMemMgr.allRegisteredBuffers_.end(),
        [parentBuf](const auto& e) {
            return e.first.get() == parentBuf;
        });
    ASSERT_NE(itParent, roceRegedMemMgr.allRegisteredBuffers_.end());
    EXPECT_TRUE(itParent->second);

    auto itChild = std::find_if(
        roceRegedMemMgr.allRegisteredBuffers_.begin(), roceRegedMemMgr.allRegisteredBuffers_.end(),
        [childHandle](const auto& e) {
            return e.first.get() == childHandle;
        });
    ASSERT_NE(itChild, roceRegedMemMgr.allRegisteredBuffers_.end());
    EXPECT_FALSE(itChild->second);

    // 再解注册子（ref 1→0，tree entry removed，子从allBuffers_擦除）
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(childHandle), HCCL_SUCCESS);
    EXPECT_FALSE(roceRegedMemMgr.localRdmaRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 0u);
    EXPECT_EQ(roceRegedMemMgr.handlesRecords_.size(), 0u);

    // 验证：子已从allBuffers_中移除
    itChild = std::find_if(
        roceRegedMemMgr.allRegisteredBuffers_.begin(), roceRegedMemMgr.allRegisteredBuffers_.end(),
        [childHandle](const auto& e) {
            return e.first.get() == childHandle;
        });
    EXPECT_EQ(itChild, roceRegedMemMgr.allRegisteredBuffers_.end());
}

// GetAllMemHandles: 空记录 → 注册 → 多次注册 → 解注册 → 验证句柄数
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_GetAllMemHandles_Expect_CorrectCount)
{
    RoceRegedMemMgr roceRegedMemMgr{nullptr};

    void* handles = nullptr;
    uint32_t count = 99U;
    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 0U);
    EXPECT_EQ(handles, nullptr);

    HcommMem mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = (void*)0x8000;
    mem.size = 4096;
    void* h1 = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t1", &h1), HCCL_SUCCESS);

    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 1U);
    EXPECT_NE(handles, nullptr);

    void* h2 = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t2", &h2), HCCL_SUCCESS);

    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 2U);

    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 1U);

    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 0U);
}
