/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_TS_URMA_CHANNEL_H
#define AICPU_TS_URMA_CHANNEL_H

#include <atomic>
#include "../channel.h"
#include "../../sockets/socket_mgr.h"

// Orion
#include "../../../../../../legacy/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/framework/resource_manager/socket/socket_manager.h"
#include "../../../../../../legacy/unified_platform/pub_inc/buffer_key.h"
#include "rma_connection.h"
#include "ub_mem_transport.h"
#include "dev_ub_connection.h"
#include "ub_local_notify.h"

namespace hcomm {

class AicpuTsUrmaChannel : public Channel {
public:
    AicpuTsUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc &channelDesc);
    ~AicpuTsUrmaChannel() override;

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t *notifyNum) const override;
    HcclResult GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags) override;
    ChannelStatus GetStatus() override;
    HcclResult GetUserRemoteMem(CommMem **remoteMem, char ***memTag, uint32_t *memNum) override;
    HcclResult UpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum) override;

    HcclResult H2DResPack(std::vector<char>& buffer);
    HcommChannelKind GetChannelKind() const override;

    HcclResult Clean() override;
    HcclResult Resume() override;

    // 数据面接口
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult Write(void *dst, const void *src, uint64_t len) override;
    HcclResult Read(void *dst, const void *src, uint64_t len) override;
    HcclResult ChannelFence() override;

private:
    HcclResult Makebufs(HcommMemHandle *memHandles, uint32_t memHandleNum, std::vector<std::shared_ptr<Hccl::Buffer>> &bufs) const;
    HcclResult ParseInputParam();
    HcclResult BuildAttr();
    HcclResult BuildConnection();
    HcclResult BuildNotify();
    HcclResult BuildBuffer(std::vector<std::shared_ptr<Hccl::Buffer>> &bufs);
    HcclResult BuildUbMemTransport();
    HcclResult BuildSocket();

    HcclResult PackOpData(std::vector<char> &data);
    HcclResult StartListen();

private:
    std::atomic<bool> isFirstPrintChannelInfo_{true}; // 是否第一次打印通道建链信息，避免重复打印日志刷屏
    // --------------------- 入参 ---------------------
    EndpointHandle                                              endpointHandle_;
    HcommChannelDesc                                            channelDesc_;

    // TODO: 成员变量全部初始化
    // --------------------- 转换参数 ---------------------
    EndpointDesc                                                localEp_{};
    EndpointDesc                                                remoteEp_{};
    uint32_t                                                    notifyNum_{0};
    std::vector<std::shared_ptr<Hccl::Buffer>>                  bufs_{};
    std::vector<std::shared_ptr<Hccl::Buffer>>                  bufsTemp{}; // channel 复用时暂存新增 buffer

    // --------------------- 具体成员 ---------------------
    Hccl::Socket*                                               socket_{nullptr};
    RdmaHandle                                                  rdmaHandle_{nullptr};
    std::unique_ptr<Hccl::UbMemTransport>                       memTransport_{nullptr};
    Hccl::BaseMemTransport::Attribution                         attr_{};
    Hccl::BaseMemTransport::CommonLocRes                        commonRes_{};
    std::vector<Hccl::LocalRmaBuffer *>                         bufferVecTemp_; // channel 复用时暂存新增 rmaBuffer
    std::vector<std::unique_ptr<Hccl::DevUbConnection>>         connections_{};
    std::vector<std::unique_ptr<Hccl::LocalUbRmaBuffer>>        localRmaBuffers_{};
    std::vector<std::unique_ptr<Hccl::UbLocalNotify>>           localNotifies_{};
    std::unique_ptr<Hccl::Socket>                               serverSocket_;
    const Hccl::SocketConfig*                                   socketConfig_{nullptr};
    uint32_t                                                    devicePhyId_{};
};

} // namespace hcomm

#endif // AICPU_TS_URMA_CHANNEL_H