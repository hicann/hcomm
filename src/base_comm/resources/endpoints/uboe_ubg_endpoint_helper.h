/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UBOE_UBG_ENDPOINT_HELPER_H
#define UBOE_UBG_ENDPOINT_HELPER_H

#include <cstdint>
#include <memory>
#include <string>
#include "endpoint.h"

namespace hcomm {
/**
 * @note 职责：UBOE/UBG协议Endpoint的公共基类，提供内存管理和Socket监听等公共实现。
 *       子类只需实现 Init() 完成各自的地址获取（UBOE做IP→EID转换，UBG直接使用EID）。
 */
class UboeUbgEndpointHelper : public Endpoint {
public:
    explicit UboeUbgEndpointHelper(const EndpointDesc& endpointDesc);
    ~UboeUbgEndpointHelper() = default;

    HcclResult ServerSocketListen(const uint32_t port) override;
    HcclResult ServerSocketStopListen(const uint32_t port) override;

    std::shared_ptr<RegedMemMgr> GetRegedMemMgr() override { return regedMemMgr_; }

    HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override;
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;
};

} // namespace hcomm

#endif // UBOE_UBG_ENDPOINT_HELPER_H
