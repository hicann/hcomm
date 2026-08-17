/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UB_RTP_ENDPOINT_H
#define UB_RTP_ENDPOINT_H

#include "uboe_ub_rtp_endpoint_helper.h"

namespace hcomm {
/**
 * @note 职责：AICPU通信引擎+UB_RTP协议的通信设备Endpoint，管理通信设备上下文，以及设备上的注册内存。
 *       UB_RTP的Init直接使用EID地址（不做IP→EID转换），rdmaHandle通过EID直接获取。
 */
class UbRtpEndpoint : public UboeUbRtpEndpointHelper {
public:
    explicit UbRtpEndpoint(const EndpointDesc& endpointDesc);
    ~UbRtpEndpoint() = default;

    HcclResult Init() override;
};

} // namespace hcomm

#endif // UB_RTP_ENDPOINT_H
