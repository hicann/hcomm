/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "comm_mems.h"
#include <cstdlib>
#include <algorithm>
#include "orion_adapter_rts.h"

namespace hccl {

CommMemType ConvertHcclToCommMemType(HcclMemType hcclType) {
    switch (hcclType) {
        case HCCL_MEM_TYPE_DEVICE:
            return COMM_MEM_TYPE_DEVICE;
        case HCCL_MEM_TYPE_HOST:
            return COMM_MEM_TYPE_HOST;
        default:
            return COMM_MEM_TYPE_INVALID;
    }
}

HcclMemType ConvertCommToHcclMemType(CommMemType commType) {
    switch (commType) {
        case COMM_MEM_TYPE_DEVICE:
            return HCCL_MEM_TYPE_DEVICE;
        case COMM_MEM_TYPE_HOST:
            return HCCL_MEM_TYPE_HOST;
        default:
            return HCCL_MEM_TYPE_NUM;
    }
}

CommMems::CommMems(uint64_t bufferSize)
    : bufferSize_(bufferSize)
{
    cclMemInfo_.mem.addr = nullptr;
    cclMemInfo_.mem.size = 0;
    cclMemInfo_.mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
}

HcclResult CommMems::Add(void *addr, uint64_t len)
{
    return HCCL_SUCCESS;
}

HcclResult CommMems::GetHcclBuffer(void *&addr, uint64_t &len)
{
    addr = reinterpret_cast<void*>(cclMemInfo_.mem.addr);
    len = static_cast<uint64_t>(cclMemInfo_.mem.size);
    return HCCL_SUCCESS;
}

HcclResult CommMems::HcclBufferMemset(void *&addr, uint64_t &len, bool clearFlag) const
{
    if (!clearFlag) {
        HCCL_DEBUG("[CommMems][HcclBufferMemset] clearFlag[%d] is false, skip memset.", clearFlag);
        return HCCL_SUCCESS;
    }

    if (addr != nullptr && len > 0) {
        EXECEPTION_CATCH(Hccl::HrtMemset(addr, len, len), return HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[CommMems][HcclBufferMemset] buffer[%p] is null or size[%llu] is 0, skip.", addr, len);
    return HCCL_E_PARA;
}

HcclResult CommMems::Init(HcclMem cclBuffer)
{
    cclMemInfo_.mem.addr = cclBuffer.addr;
    cclMemInfo_.mem.size = cclBuffer.size;
    cclMemInfo_.mem.type = ConvertHcclToCommMemType(cclBuffer.type);
    std::string memTag = "HcclBuffer";
    errno_t sRet = strncpy_s(cclMemInfo_.memTag, HCOMM_RES_TAG_MAX_LEN, memTag.c_str(), memTag.size());
    CHK_PRT_RET(sRet != EOK,
        HCCL_ERROR("[CommRegMem] strncpy_s failed, return [%d].", sRet), HCCL_E_MEMORY);
    HCCL_INFO("[CommMems][Init] addr[%p] size[%u] memType[%u]", cclBuffer.addr, cclBuffer.size, cclBuffer.type);
    return HCCL_SUCCESS;
}

HcclResult CommMems::GetMemoryHandles(std::vector<HcclMem> &mem)
{
    HcclMem memTemp;
    memTemp.size = cclMemInfo_.mem.size;
    memTemp.type = ConvertCommToHcclMemType(cclMemInfo_.mem.type);
    memTemp.addr = cclMemInfo_.mem.addr;
    mem.push_back(memTemp);

    HCCL_INFO("[CommMems][%s] HcclMem: size[%u], addr[%p], type[%d]", 
        __func__, memTemp.size, memTemp.addr, (int)memTemp.type
    );

    return HCCL_SUCCESS;
}

HcclResult CommMems::CommRegMem(const std::string& memTag, const CommMem& mem,
    void **memHandle)
{
    CHK_PRT_RET(memHandle == nullptr, HCCL_ERROR("[CommRegMem] memHandle is null. tag[%s]", memTag.c_str()), HCCL_E_PARA);
    CHK_PRT_RET(mem.addr == nullptr || mem.size == 0, HCCL_ERROR("[CommRegMem] invalid mem. addr[%p] size[%llu]",
        mem.addr, (unsigned long long)mem.size), HCCL_E_PARA);
    if (UNLIKELY(memTag.size() >= HCOMM_RES_TAG_MAX_LEN)) {
        HCCL_ERROR("[CommRegMem] memTag.size() exceeds limit[%u]", HCOMM_RES_TAG_MAX_LEN);
        return HCCL_E_PARA;
    }

    // 组装句柄（仅域内管理，无进程级注册）
    Handle h;
    EXECEPTION_CATCH(h = std::make_shared<CommMemInfo>(), return HCCL_E_PTR);
    h->mem.addr    = mem.addr;
    h->mem.size    = mem.size;
    h->mem.type    = mem.type;
    errno_t sRet = strncpy_s(h->memTag, HCOMM_RES_TAG_MAX_LEN, memTag.c_str(), memTag.size());
    CHK_PRT_RET(sRet != EOK,
        HCCL_ERROR("[CommRegMem] strncpy_s failed, return [%d].", sRet), HCCL_E_MEMORY);

    const auto key = MakeKey(mem.addr, static_cast<size_t>(mem.size));
 
    std::lock_guard<std::mutex> addLock(memMutex_);

    auto opIt = opBindings_.find(memTag);
    if (opIt != opBindings_.end()) {
        HCCL_ERROR("[CommRegMem] memTag[%s] has already been registered.", memTag.c_str());
        return HCCL_E_PARA;
    }

    auto& reg = tagRegs_[memTag];
 
    // 同tag内做区间冲突/幂等复用
    reg.table.AddWithoutCheck(key, h);

    // 加入绑定map
    opBindings_.emplace(memTag, h);

    // 新增反查map，用于aiv建链交换内存
    auto opRevIt = opReverseBindings_.find(h.get());
    if (opRevIt == opReverseBindings_.end()) {
        opReverseBindings_[h.get()] = memTag;
    }
 
    *memHandle = h.get();
    HCCL_INFO("[CommRegMem] ok. tag[%s] memHandle[%p] size[%llu]", memTag.c_str(), *memHandle,
        (unsigned long long)h->mem.size);
    return HCCL_SUCCESS;
}
 
HcclResult CommMems::CommUnregMem(const std::string& memTag, const void* memHandle) // 待确认是否要解注册
{
    CHK_PRT_RET(memHandle == nullptr, HCCL_ERROR("[CommUnregMem] memHandle is null"), HCCL_E_PARA);
    CHK_PRT_RET(memTag.empty(), HCCL_ERROR("[CommUnregMem] memTag is null or empty"), HCCL_E_PARA);
 
    std::lock_guard<std::mutex> addLock(memMutex_);
 
    auto itTag = opBindings_.find(memTag);
    CHK_PRT_RET(itTag == opBindings_.end(),
        HCCL_WARNING("[CommUnregMem] tag[%s] not found in bindings", memTag.c_str()), HCCL_E_NOT_FOUND);
 
    auto &h = itTag->second;           // Handle under this tag
    auto &reg = tagRegs_[itTag->first];       // TagRegistry for this tag
    size_t unboundCount = 0;  // 本次解绑命中的句柄个数（即便 Del 未真正擦除也计数）
    size_t erasedCount  = 0;  // RmaBufferMgr::Del 返回 true 的次数（ref 归零而“擦除”）
    
    if (h.get() == memHandle) {
        const auto key = MakeKey(h->mem.addr, static_cast<size_t>(h->mem.size));
        try {
            if (reg.table.Del(key)) {
                ++erasedCount;            // 该 key 的引用归零并从表中移除
            }
        } catch (const std::out_of_range &) {
            HCCL_ERROR("[CommUnregMem] tag[%s] key not found on Del (maybe already removed)", itTag->first.c_str());
        }
        ++unboundCount;                   // 从绑定列表移除，无论 Del 是否真正擦除
        opReverseBindings_.erase(const_cast<void*>(memHandle)); // 这里考虑增加校验
        opBindings_.erase(itTag);
        if (reg.table.size() == 0) {
            tagRegs_.erase(std::string(memTag));
        }
    }
 
    CHK_PRT_RET(unboundCount == 0,
        HCCL_WARNING("[CommUnregMem] tag[%s] memHandle[%p] not found", memTag.c_str(), memHandle), HCCL_E_NOT_FOUND);
 
    HCCL_INFO("[CommUnregMem] tag[%s] memHandle[%p] unbound=%zu, erased=%zu",
              memTag.c_str(), memHandle, unboundCount, erasedCount);
    return HCCL_SUCCESS;
}
 
HcclResult CommMems::GetTagMemoryHandles(void** memHandles, uint32_t memHandleNum, std::vector<HcclMem> &memVec, 
    std::vector<std::string> &memTag)
{
    HcclMem memTemp;
    memTemp.size = cclMemInfo_.mem.size;
    memTemp.type = ConvertCommToHcclMemType(cclMemInfo_.mem.type);
    memTemp.addr = cclMemInfo_.mem.addr;
    memVec.push_back(memTemp);
    memTag.push_back("HcclBuffer");
 
    // 增加入参检查
    std::lock_guard<std::mutex> lock(memMutex_);
    CommMemInfo** handles = reinterpret_cast<CommMemInfo**>(memHandles);
    for (uint32_t i = 0; i < memHandleNum; i++) {
        auto it = opReverseBindings_.find(handles[i]);
        if (it == opReverseBindings_.end()) {
            HCCL_ERROR("[CommMems] memHandle[%p] not found", handles[i]);
            return HCCL_E_NOT_FOUND;
        }
        HcclMem mem;
        mem.addr = (*handles[i]).mem.addr;
        mem.size = (*handles[i]).mem.size;
        mem.type = ConvertCommToHcclMemType((*handles[i]).mem.type);
        memTag.push_back(opReverseBindings_[handles[i]]);
        memVec.push_back(mem);
    }
    return HCCL_SUCCESS;
}

HcclResult CommMems::SetMemHandles(HcommMemHandle *memHandles, const std::vector<MemHandle> &memHandleVec,
    std::vector<MemHandle> &commMemHandleVec)
{
    if (memHandleVec.size() == 0) {
        HCCL_ERROR("[CommMems][SetMemHandles] memHandleVecSize is 0.");
        return HCCL_E_PARA;
    }
    CHK_PTR_NULL(memHandleVec[0]);
    cclMemInfo_.bufferHandle = memHandleVec[0];
    commMemHandleVec.push_back(static_cast<void*>(&cclMemInfo_));

    CommMemInfo **handles = reinterpret_cast<CommMemInfo**>(memHandles);
    for (uint32_t i = 1; i < memHandleVec.size(); ++i) {
        CHK_PTR_NULL(memHandleVec[i]);
        (*handles[i - 1]).bufferHandle = memHandleVec[i];
        commMemHandleVec.push_back(static_cast<void*>(handles[i - 1]));
    }
    return HCCL_SUCCESS;
}
}