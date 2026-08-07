/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "endpoint_pair.h"
#include "log.h"
#include "roce_mem.h"
#include "exchange_rdma_buffer_dto.h"
#include "local_rdma_rma_buffer.h"
#include "hccl_one_sided_data.h"

namespace hcomm {

RoceRegedMemMgr::RoceRegedMemMgr() { localRdmaRmaBufferMgr_ = std::make_unique<LocalRdmaRmaBufferMgr>(); }

HcclResult RoceRegedMemMgr::RegisterMemory(HcommMem mem, const char* memTag, void** memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(localRdmaRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return RegisterMemoryImpl(
        mem, memTag, memHandle, localRdmaRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_, "RoceRegedMemMgr",
        [&](auto& bufPtr, auto& parent) {
            return std::make_shared<Hccl::LocalRdmaRmaBuffer>(
                bufPtr, rdmaHandle_, parent->GetLkey(), parent->GetRkey(), parent->GetMrHandle());
        },
        [&](auto& bufPtr) {
            return std::make_shared<Hccl::LocalRdmaRmaBuffer>(bufPtr, rdmaHandle_);
        });
}

HcclResult RoceRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(localRdmaRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return UnregisterMemoryImpl(
        memHandle, localRdmaRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_,
        [](auto* b) {
            return b->GetLkey();
        },
        [](auto a, auto b) {
            return a == b;
        });
}

HcclResult RoceRegedMemMgr::GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalRdmaRmaBuffer* localRdmaRmaBuffer)
{
    auto dto = localRdmaRmaBuffer->GetExchangeDto();
    Hccl::BinaryStream localRdmaRmaBufferStream;
    dto->Serialize(localRdmaRmaBufferStream);
    std::vector<char> tempLocalMemDesc;
    localRdmaRmaBufferStream.Dump(tempLocalMemDesc);
    HCCL_DEBUG("[RoceRegedMemMgr][GetMemDesc] [%s] dump data size [%u]", __func__, tempLocalMemDesc.size());
    // 判断内存描述符是否正确导出
    if (tempLocalMemDesc.empty()) {
        HCCL_ERROR("[RoceRegedMemMgr][GetMemDesc] [%s] tempLocalMemDesc export failed.", __func__);
        return HCCL_E_INTERNAL;
    }

    std::vector<char> tempLocalEndpointDesc;
    tempLocalEndpointDesc.resize(sizeof(EndpointDesc));
    if (memcpy_s(tempLocalEndpointDesc.data(), sizeof(EndpointDesc), &endpointDesc, sizeof(EndpointDesc)) != EOK) {
        HCCL_ERROR("[RoceRegedMemMgr][GetMemDesc] [%s] endpointDesc memcpy_s failed.", __func__);
        return HCCL_E_INTERNAL;
    }

    tempLocalMemDesc.insert(tempLocalMemDesc.end(), tempLocalEndpointDesc.begin(), tempLocalEndpointDesc.end());

    // 内存描述符拷贝
    localRdmaRmaBuffer->Desc = std::move(tempLocalMemDesc);
    return HCCL_SUCCESS;
}

HcclResult
RoceRegedMemMgr::MemoryExport(const EndpointDesc endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(memDescLen);
    std::lock_guard<std::mutex> lock(memMtx_);

    Hccl::LocalRdmaRmaBuffer* localRdmaRmaBuffer = nullptr;
    CHK_RET(ValidateMemExportHandle(memHandle, allRegisteredBuffers_, localRdmaRmaBuffer));

    // 获取序列化信息
    CHK_RET(GetMemDesc(endpointDesc, localRdmaRmaBuffer));

    *memDescLen = static_cast<uint32_t>(localRdmaRmaBuffer->Desc.size());
    *memDesc = static_cast<void*>(localRdmaRmaBuffer->Desc.data());
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::GetParamsFromMemDesc(
    const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, Hccl::ExchangeRdmaBufferDto& dto)
{
    const char* description = static_cast<const char*>(memDesc);

    if (descLen < sizeof(EndpointDesc)) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][GetParamsFromMemDesc] [%s] descLen[%u] is too small. aim size:[%llu]", __func__, descLen,
            sizeof(EndpointDesc));
        return HCCL_E_INTERNAL;
    }
    // 从memDesc末尾提取EndpointDesc
    if (memcpy_s(
            &endpointDesc, sizeof(EndpointDesc), description + descLen - sizeof(EndpointDesc), sizeof(EndpointDesc))
        != EOK) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][GetParamsFromMemDesc] [%s] endpointDesc copy error. aim size:[%llu]", __func__,
            sizeof(EndpointDesc));
        return HCCL_E_INTERNAL;
    }

    // 反序列化
    std::vector<char> tempDesc{};
    tempDesc.resize(TRANSPORT_EMD_ESC_SIZE);
    tempDesc.assign(description, description + descLen - sizeof(EndpointDesc));
    Hccl::BinaryStream remoteRdmaRmaBufferStream(tempDesc);
    dto.Deserialize(remoteRdmaRmaBufferStream);
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    std::lock_guard<std::mutex> lock(memMtx_);

    EndpointDesc endpointDesc;
    Hccl::ExchangeRdmaBufferDto dto;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, dto));

    // 构造RemoteRdmaRmaBuffer
    std::shared_ptr<Hccl::RemoteRdmaRmaBuffer> remoteRdmaRmaBuffer;
    EXCEPTION_CATCH(remoteRdmaRmaBuffer = std::make_shared<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle_, dto),
                    return HCCL_E_PTR;);

    // 放到RemoteRdmaRmaBufferMgr_
    hccl::BufferKey<uintptr_t, u64> tempKey(static_cast<uintptr_t>(dto.addr), dto.size);
    if (remoteRdmaRmaBufferMgrs_.find(endpointDesc) == remoteRdmaRmaBufferMgrs_.end()) {
        std::unique_ptr<RemoteRdmaRmaBufferMgr> remoteRdmaRmaBufferMgr;
        EXCEPTION_CATCH((remoteRdmaRmaBufferMgr = std::make_unique<RemoteRdmaRmaBufferMgr>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(remoteRdmaRmaBufferMgr);
        remoteRdmaRmaBufferMgrs_[endpointDesc] = std::move(remoteRdmaRmaBufferMgr);
        HCCL_INFO("remoteRdmaRmaBufferMgrs_ add remoteRdmaRmaBufferMgr successfully!");
    }

    auto resultPair = remoteRdmaRmaBufferMgrs_[endpointDesc]->Add(tempKey, remoteRdmaRmaBuffer);
    if (!resultPair.second) {
        HCCL_ERROR("[RoceRegedMemMgr][MemoryImport] This memDesc has already been imported!");
        return HCCL_E_AGAIN;
    }

    outMem->addr = reinterpret_cast<void*>(remoteRdmaRmaBuffer->GetAddr());
    outMem->size = remoteRdmaRmaBuffer->GetSize();

    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    std::lock_guard<std::mutex> lock(memMtx_);

    EndpointDesc endpointDesc;
    Hccl::ExchangeRdmaBufferDto dto;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, dto));

    if (remoteRdmaRmaBufferMgrs_.find(endpointDesc) == remoteRdmaRmaBufferMgrs_.end()) {
        HCCL_ERROR("[RoceRegedMemMgr][MemoryUnimport] Remote buffer manager Not Found.");
        return HCCL_E_NOT_FOUND;
    }

    // 删除RemoteRdmaRmaBuffer
    HCCL_INFO("[MemoryUnimport][Rdma] MemoryUnimport");
    hccl::BufferKey<uintptr_t, u64> tempKey(static_cast<uintptr_t>(dto.addr), dto.size);

    bool resultPair = false;
    EXCEPTION_CATCH(resultPair = remoteRdmaRmaBufferMgrs_[endpointDesc]->Del(tempKey), return HCCL_E_NOT_FOUND);
    // 计数器大于1时，返回false，说明框架层有其它设备在使用这段内存，返回HCCL_E_AGAIN
    if (!resultPair) {
        HCCL_INFO("[RoceRegedMemMgr][[MemoryUnimport] Memory reference count is larger than 0"
                  "(used by other RemoteRank).");
        return HCCL_E_AGAIN;
    }
    if (!remoteRdmaRmaBufferMgrs_[endpointDesc]->size()) {
        remoteRdmaRmaBufferMgrs_.erase(endpointDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    std::lock_guard<std::mutex> lock(memMtx_);
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);
    *memHandleNum = static_cast<uint32_t>(handlesRecords_.size());
    *memHandles = handlesRecords_.empty() ? nullptr : static_cast<void*>(handlesRecords_.data());
    HCCL_INFO("[RoceRegedMemMgr][GetAllMemHandles] memHandleNum[%u]", *memHandleNum);
    return HCCL_SUCCESS;
}

} // namespace hcomm
