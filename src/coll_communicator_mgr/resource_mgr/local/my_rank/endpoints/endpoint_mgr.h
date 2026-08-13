/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_MGR_H
#define ENDPOINT_MGR_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "endpoint.h"
#include "../endpoint_pairs/endpoint_pair.h"
#include "hccl_mem_defs.h"

namespace hcomm {

/**
 * @brief endpoint 粒度的 tag→handle 映射。
 *        持有 EndpointHandle，析构时自动解注册全部 tag 对应内存。
 */
class TaggedMemMap {
public:
    explicit TaggedMemMap(EndpointHandle handle) : handle_(handle) {}
    ~TaggedMemMap();

    TaggedMemMap(const TaggedMemMap&) = delete;
    TaggedMemMap& operator=(const TaggedMemMap&) = delete;
    TaggedMemMap(TaggedMemMap&&) = default;
    TaggedMemMap& operator=(TaggedMemMap&&) = default;

    uint64_t GetVersion() const { return version_; }
    void SetVersion(uint64_t ver) { version_ = ver; }

    MemHandle FindHandle(const std::string& tag) const;
    bool HasTag(const std::string& tag) const;
    void EmplaceHandle(const std::string& tag, MemHandle handle);
    MemHandle RemoveTag(const std::string& tag);

private:
    EndpointHandle handle_;
    uint64_t version_{0};
    std::unordered_map<std::string, MemHandle> tagToHandle_;
};

/**
 * @note 职责：Endpoint管理器，支持不同类型的Endpoint的创建和销毁管理。
 */
class EndpointMgr {
public:
    EndpointMgr() {};
    ~EndpointMgr();

    // 获取端点
    HcclResult Get(EndpointDesc epDesc, EndpointHandle& handle);

    // 注册内存到端点，若 commMemsVersion 与上次注册时一致则跳过
    HcclResult RegisterMemory(
        EndpointHandle epHandle, const std::vector<std::string>& memTag, const std::vector<HcclMem>& memVec,
        uint64_t commMemsVersion);

    // 查询指定 endpoint 下若干 tag 对应的 MemHandle
    HcclResult GetMemHandlesByTags(
        EndpointHandle epHandle, const std::vector<std::string>& memTags, std::vector<MemHandle>& memHandleVec);

    // 从所有 endpoint 中删除指定 tag 并调 HcommMemUnreg
    HcclResult UnregMemByTag(const std::string& tag);

private:
    bool IsDescExist(EndpointDesc epDesc);

private:
    std::unordered_map<EndpointDesc, EndpointHandle> endpointMap_{};
    std::unordered_map<EndpointHandle, TaggedMemMap> endpointTagMemMap_{};
    std::mutex mutex_{};
};

} // namespace hcomm

#endif // ENDPOINT_MGR_H
