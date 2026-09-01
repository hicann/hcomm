/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SERVER_SOCKET_CONTEXT_H
#define SERVER_SOCKET_CONTEXT_H

#include <cstdint>
#include "port.h"
#include "ip_address.h"

namespace hcomm {

/**
 * @note 职责：封装 ServerSocketListen/ServerSocketStopListen/ServerSocketGetListenPort。
 *       基类形态与迁移前 Endpoint 基类一致：Listen 纯虚；StopListen/GetListenPort 虚默认返回
 *       HCCL_E_NOT_SUPPORT（迁移前未覆写的子类继承此行为）。无公共数据成员
 *       （三种路径差异大，protoType_/dynamicPort_ 等公共成员下放到各子类）。
 *       ipAddr 为每次监听的目标地址，由调用方从 endpointDesc.commAddr 解析后传入
 *       （HcclSocket 自管理/GlobalNetDevMgr 路径不使用 ipAddr，签名保持与 Host/Device 子类一致）。
 */
class ServerSocketContext {
public:
    ServerSocketContext() = default;
    virtual ~ServerSocketContext() = default;

    virtual HcclResult ServerSocketListen(const Hccl::IpAddress& ipAddr, uint32_t port) = 0;
    virtual HcclResult
    ServerSocketStopListen([[maybe_unused]] const Hccl::IpAddress& ipAddr, [[maybe_unused]] const uint32_t port)
    {
        return HCCL_E_NOT_SUPPORT;
    }
    virtual HcclResult
    ServerSocketGetListenPort([[maybe_unused]] const Hccl::IpAddress& ipAddr, [[maybe_unused]] uint32_t* port)
    {
        return HCCL_E_NOT_SUPPORT;
    }
};

} // namespace hcomm

#endif // SERVER_SOCKET_CONTEXT_H
