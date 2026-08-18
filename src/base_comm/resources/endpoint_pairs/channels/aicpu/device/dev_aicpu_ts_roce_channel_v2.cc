/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dev_aicpu_ts_roce_channel_v2.h"
#include "log.h"
#include "aicpu_res_package_helper.h"

namespace Hccl {

DevAicpuTsRoceChannelV2::DevAicpuTsRoceChannelV2(std::vector<char>& uniqueId)
{
    transport_ = std::make_unique<RoceTransportLiteImpl>(uniqueId);
}

DevAicpuTsRoceChannelV2::~DevAicpuTsRoceChannelV2() {}

std::string DevAicpuTsRoceChannelV2::Describe() const
{
    if (transport_ != nullptr) {
        return transport_->Describe();
    }
    return "DevAicpuTsRoceChannelV2[transport=nullptr]";
}

HcclResult DevAicpuTsRoceChannelV2::Create(
    const void* blob, u64 blobBytes, [[maybe_unused]] const HcommDeviceInfo& deviceInfo, ChannelHandle& outHandle)
{
    CHK_PTR_NULL(blob);
    CHK_PRT_RET(blobBytes == 0, HCCL_ERROR("[DevAicpuTsRoceChannelV2][Create] blobBytes is 0"), HCCL_E_PARA);

    std::vector<char> data(blobBytes);
    CHK_SAFETY_FUNC_RET(memcpy_s(data.data(), data.size(), blob, blobBytes));

    Hccl::AicpuResPackageHelper helper;
    auto dataVec = helper.ParsePackedData(data);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    if (static_cast<u32>(resType) >= dataVec.size()) {
        HCCL_ERROR(
            "[DevAicpuTsRoceChannelV2][%s] fail, resType[%d], dataVec size[%u]", __func__, resType, dataVec.size());
        return HCCL_E_PARA;
    }

    transport_ = std::make_unique<RoceTransportLiteImpl>();
    CHK_PTR_NULL(transport_);
    transport_->Init(dataVec[resType].data);

    outHandle = reinterpret_cast<ChannelHandle>(transport_.get());

    HCCL_INFO(
        "[DevAicpuTsRoceChannelV2][Create] success blobBytes[%llu] handle[0x%llx]",
        static_cast<unsigned long long>(blobBytes),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(outHandle)));
    return HCCL_SUCCESS;
}

bool DevAicpuTsRoceChannelV2::Destroy(ChannelHandle handle)
{
    if (transport_ != nullptr && reinterpret_cast<ChannelHandle>(transport_.get()) == handle) {
        transport_.reset();
    } else {
        auto* transport = reinterpret_cast<RoceTransportLiteImpl*>(handle);
        delete transport;
    }
    HCCL_DEBUG("[DevAicpuTsRoceChannelV2][Destroy] destroyed handle[0x%llx]", handle);
    return true;
}

} // namespace Hccl
