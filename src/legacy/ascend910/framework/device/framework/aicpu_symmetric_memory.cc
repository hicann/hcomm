/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_symmetric_memory.h"
#include "symmetric_memory/symmetric_memory.h"

#ifdef CCL_KERNEL_AICPU
namespace hccl {

class SymmetricMemory::SimpleVaAllocator {
public:
    // 不需要任何成员，只要让编译器觉得它是个完整的类就行
    SimpleVaAllocator() {}
    ~SimpleVaAllocator() {}
};

SymmetricMemory::~SymmetricMemory() {}
} // namespace hccl
#endif // CCL_KERNEL_AICPU

using namespace hccl;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult HcclSymWinGetPeerPointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    CHK_PTR_NULL(winHandle);
    CHK_PTR_NULL(ptr);
    SymmetricWindow* symWin = static_cast<SymmetricWindow*>(winHandle);
    if (symWin->mode == SymmetricMemoryMode::URMA) {
        HCCL_ERROR("[HcclSymWinGetPeerPointer] not support URMA mode, winHandle[%p] is URMA mode.", winHandle);
        *ptr = nullptr;
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        peerRank >= symWin->rankSize,
        HCCL_ERROR("[HcclSymWinGetPeerPointer] Invalid peerRank: %d. rankSize[%u]", peerRank, symWin->rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(
        offset >= symWin->userSize,
        HCCL_ERROR("[%s] Invalid offset: %llu. userSize[%llu]", __func__, offset, symWin->userSize), HCCL_E_PARA);

    size_t peerOffset = peerRank * symWin->stride + offset;
    *ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(symWin->baseVa) + peerOffset);
    HCCL_INFO(
        "[HcclSymWinGetPeerPointer] Get Ptr[%p] from winHandle[%p], peerRank[%d], peerOffset[%llu]", *ptr, winHandle,
        peerRank, peerOffset);

    return HCCL_SUCCESS;
}

HcclResult HcclSymWinGetRemoteAddr(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    CHK_PTR_NULL(winHandle);
    CHK_PTR_NULL(ptr);
    SymmetricWindow* symWin = static_cast<SymmetricWindow*>(winHandle);
    if (symWin->mode != SymmetricMemoryMode::URMA) {
        HCCL_ERROR("[HcclSymWinGetRemoteAddr] only support URMA mode, winHandle[%p] is not URMA mode.", winHandle);
        *ptr = nullptr;
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(symWin->remoteMems);
    CHK_PRT_RET(
        peerRank >= symWin->remoteMemNum,
        HCCL_ERROR("[HcclSymWinGetRemoteAddr] Invalid peerRank: %d. remoteMemNum[%u]", peerRank, symWin->remoteMemNum),
        HCCL_E_PARA);
    CommMem& remoteMem = symWin->remoteMems[peerRank];
    CHK_PRT_RET(
        remoteMem.addr == nullptr || remoteMem.type == COMM_MEM_TYPE_INVALID,
        HCCL_ERROR(
            "[HcclSymWinGetRemoteAddr] remote mem is invalid, peerRank[%u], addr[%p], type[%d].", peerRank,
            remoteMem.addr, remoteMem.type),
        HCCL_E_PARA);
    CHK_PRT_RET(
        offset >= remoteMem.size,
        HCCL_ERROR("[%s] Invalid offset: %llu. remoteMemSize[%llu]", __func__, offset, remoteMem.size), HCCL_E_PARA);
    *ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(remoteMem.addr) + offset);
    HCCL_INFO(
        "[HcclSymWinGetRemoteAddr] Get URMA Ptr[%p] from winHandle[%p], peerRank[%d], offset[%llu]", *ptr, winHandle,
        peerRank, offset);

    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
