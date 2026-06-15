/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_ub_rma_buffer.h"

#include "sal.h"
#include "rma_buffer.h"
#include "null_ptr_exception.h"
#include "exchange_ub_buffer_dto.h"
#include "rdma_handle_manager.h"
#include "inner_net_dev_manager.h"

namespace Hccl {

constexpr u32 TEN_MILLISECOND_OF_USLEEP = 10000;

LocalUbRmaBuffer::LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle)
    : LocalRmaBuffer(buf, RmaType::UB), rdmaHandle(rdmaHandle)
{
    if (rdmaHandle == nullptr) {
        THROW<NullPtrException>("LocalUbRmaBuffer's rdmaHandle is nullptr");
    }
    std::pair<u64, u64> alignBuf = BufAlign(buf->GetAddr(), buf->GetSize());

    const auto &tokenIdInfoPair = RdmaHandleManager::GetInstance().GetTokenIdInfo(rdmaHandle, 
        BufferKey<uintptr_t, u64>{alignBuf.first, alignBuf.second});
    tokenIdHandle = tokenIdInfoPair.first;
    tokenId       = tokenIdInfoPair.second;
    tokenValue    = GetUbToken();
    HrtRaUbLocMemRegParam lmemReg{alignBuf.first, alignBuf.second, tokenValue, tokenIdHandle, 1};
    reqReg     = HrtRaUbLocalMemReg(rdmaHandle, lmemReg);
    keySize    = reqReg.keySize;
    memHandle  = reqReg.handle;
    segVa      = reqReg.targetSegVa;
    memcpy_s(key, HRT_UB_MEM_KEY_MAX_LEN, reqReg.key, HRT_UB_MEM_KEY_MAX_LEN);

    HCCL_INFO("[LocalUbRmaBuffer::%s] end, rdmaHandle[%p], lmemHandle[0x%llx], keySize[%u]", __func__, rdmaHandle,
               memHandle, keySize);
}

LocalUbRmaBuffer::LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, void *netDevice, bool flag)
    : LocalRmaBuffer(buf, RmaType::UB)
{
    (void)flag;
    if (netDevice == nullptr) {
        THROW<NullPtrException>("LocalUbRmaBuffer's netDevice is nullptr");
    }
    tokenValue = GetUbToken();
    netDev     = reinterpret_cast<HcclNetDevice *>(netDevice);
    rdmaHandle = netDev->GetRdmaHandle();

    std::pair<u64, u64> alignBuf = BufAlign(buf->GetAddr(), buf->GetSize());

    const auto &tokenIdInfoPair = netDev->GetTokenIdInfo(BufferKey<uintptr_t, u64>{alignBuf.first, alignBuf.second});
    tokenIdHandle               = tokenIdInfoPair.first;
    tokenId                     = tokenIdInfoPair.second;
    tokenValue                  = GetUbToken();
    HrtRaUbLocMemRegParam lmemReg{alignBuf.first, alignBuf.second, tokenValue, tokenIdHandle, 1};
    reqReg    = HrtRaUbLocalMemReg(rdmaHandle, lmemReg);
    keySize   = reqReg.keySize;
    memHandle = reqReg.handle;
    memcpy_s(key, HRT_UB_MEM_KEY_MAX_LEN, reqReg.key, HRT_UB_MEM_KEY_MAX_LEN);
    HCCL_INFO("[LocalUbRmaBuffer::%s] end, rdmaHandle[%p], lmemHandle[0x%llx], keySize[%u]", __func__, rdmaHandle,
              memHandle, keySize);
}

LocalUbRmaBuffer::LocalUbRmaBuffer(std::shared_ptr<Buffer> buf) : LocalRmaBuffer(buf, RmaType::UB), rdmaHandle(nullptr)
{
    rtMemUbTokenInfo info;
    info.va   = buf->GetAddr();
    info.size = buf->GetSize();
    HrtUbDevQueryInfo(QUERY_PROCESS_TOKEN, &info);
    tokenId    = info.tokenId;
    tokenValue = info.tokenValue; // 未处理tokenIdHandle
    HCCL_INFO("LocalUbRmaBuffer Construct: buf=[%s]", buf->Describe().c_str());
}

string LocalUbRmaBuffer::Describe() const
{
    return StringFormat("LocalUbRmaBuffer[rdmaHandle=%p, buf=%s, memHandle=%p]",
                        rdmaHandle, buf->Describe().c_str(), memHandle);
}

std::unique_ptr<Serializable> LocalUbRmaBuffer::GetExchangeDto()
{
    std::unique_ptr<ExchangeUbBufferDto> dto = make_unique<ExchangeUbBufferDto>(buf->GetAddr(),
        buf->GetSize(),
        buf->GetMemType(),
        buf->GetMemTag().c_str(),
        tokenValue,
        tokenId,
        keySize);
    (void)memcpy_s(dto->key, HRT_UB_MEM_KEY_MAX_LEN, key, HRT_UB_MEM_KEY_MAX_LEN);
    dto->segVa = segVa;
    return std::unique_ptr<Serializable>(dto.release());
}

LocalUbRmaBuffer::~LocalUbRmaBuffer()
{
    if (rdmaHandle != nullptr && memHandle != 0) {
        HCCL_INFO("[LocalUbRmaBuffer::%s] rdmaHandle[%p], lmemHandle[0x%llx]", __func__, rdmaHandle, memHandle);
        DECTOR_TRY_CATCH("LocalUbRmaBuffer", HrtRaUbLocalMemUnreg(rdmaHandle, memHandle));
    }
}

u32 LocalUbRmaBuffer::GetTokenId() const
{
    return tokenId;
}

u32 LocalUbRmaBuffer::GetTokenValue() const
{
    return tokenValue;
}

TokenIdHandle LocalUbRmaBuffer::GetTokenIdHandle() const
{
    return tokenIdHandle;
}

static bool isInitialized = false;  // 标记是否已经初始化
static u32 token = 0;  // 存储生成的随机数
static std::mutex ubTokenMutex;
u32 GetUbToken()
{
    std::lock_guard<std::mutex> lock(ubTokenMutex);
    if (!isInitialized) {
        s32 devLogicId = HrtGetDevice();
        u32 devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);
        HrtRaGetSecRandom(&token, devPhyId);
        isInitialized = true;
    }
    return token;
}


} // namespace Hccl
