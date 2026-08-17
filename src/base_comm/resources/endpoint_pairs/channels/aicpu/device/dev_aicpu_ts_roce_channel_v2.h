/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_AICPU_TS_ROCE_CHANNEL_V2_H
#define DEV_AICPU_TS_ROCE_CHANNEL_V2_H

#include <vector>
#include <memory>
#include "dev_aicpu_ts_channel.h"
#include "roce_transport_lite_impl.h"

namespace Hccl {

MAKE_ENUM(RDMA_NOTIFY_TYPE, DATA_NOTIFY_MEM, ACK_NOTIFY_MEM, DATA_ACK_NOTIFY_MEM, NOTIFY_TYPE_RESERVED);

class DevAicpuTsRoceChannelV2 : public ::DevAicpuTsChannel {
public:
    explicit DevAicpuTsRoceChannelV2(std::vector<char>& uniqueId);
    DevAicpuTsRoceChannelV2() = default;
    ~DevAicpuTsRoceChannelV2() override;

    HcclResult
    Create(const void* blob, u64 blobBytes, const HcommDeviceInfo& deviceInfo, ChannelHandle& outHandle) override;
    bool Destroy(ChannelHandle handle) override;

    std::string Describe() const;

private:
    std::unique_ptr<RoceTransportLiteImpl> transport_{};
};

} // namespace Hccl
#endif
