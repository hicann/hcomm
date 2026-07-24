/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tokenInfo_manager.h"
#include "orion_adapter_rts.h"

namespace Hccl {

TokenInfo TokenInfoManager::GetTokenInfo(const BufferKey<uintptr_t, u64> &bufKey)
{
    std::lock_guard<std::mutex> lock(tokenInfoMgrMutex_);

    BufKeyVecIndex index = GetBufferVecIndex(bufKey);
    auto iter = tokenInfoMap_.find(index);
    if (iter == tokenInfoMap_.end()) {
        TokenInfo tokenInfo = RaUbAllocTokenIdHandle(rdmahandle_);
        tokenInfoMap_.emplace(index, tokenInfo);
    }
    HCCL_INFO("[TokenInfoManager::%s] rdmahandle[%p] index[%u]", __func__, rdmahandle_, index);
    return tokenInfoMap_[index];
}

bool HasIntersect(const vector<BufferKey<uintptr_t, u64>> &bufKeys, const BufferKey<uintptr_t, u64> &inputBufKey)
{
    // 遍历bufKeys, 若bufKeys中存在和inputBufKey相交的bufKey则返回true, 否则fasle
    auto it = std::find_if(bufKeys.begin(), bufKeys.end(), 
                           [inputBufKey](const auto &curBufKeyInVec) { 
                                return inputBufKey.IsIntersect(curBufKeyInVec);
                           });
    return it != bufKeys.end();
}

BufKeyVecIndex TokenInfoManager::GetBufferVecIndex(const BufferKey<uintptr_t, u64> &inputBufKey)
{
    auto &bufferKeys = bufferKeysMap_[devId_];
    u32 size = bufferKeys.size();
    auto it = std::find_if(bufferKeys.begin(), bufferKeys.end(), 
                           [inputBufKey](const auto &unOverlapBufVec) { 
                                return !HasIntersect(unOverlapBufVec, inputBufKey); 
                            });

    // 若不存在一组bufferKey与inputBufKey不相交, 需要申请新的token, 返回新的索引
    u32 idx = std::distance(bufferKeys.begin(), it);
    if (idx == size) {
        bufferKeys.push_back(vector<BufferKey<uintptr_t, u64>>{inputBufKey});
    } else {
        // 若存在一组bufferKey与inputBufKey不相交则返回该组索引, 然后将inputBufKey插入
        it->push_back(inputBufKey);
    }

    HCCL_INFO("[TokenInfoManager::%s] idx[%u] size[%u]", __func__, idx, size);
    return idx;
}

void TokenInfoManager::Destroy()
{
    HCCL_INFO("[TokenInfoManager::%s] rdmahandle[%p]", __func__, rdmahandle_);
    for (auto &tokenInfo : tokenInfoMap_) {
        DECTOR_TRY_CATCH("token id handle destroy",
                         RaUbFreeTokenIdHandle(rdmahandle_, tokenInfo.second.first));
    }
}

} // namespace Hccl
