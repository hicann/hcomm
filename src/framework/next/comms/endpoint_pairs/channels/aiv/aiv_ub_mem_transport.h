/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_UB_MEM_TRANSPORT_H
#define AIV_UB_MEM_TRANSPORT_H
#include <mutex>
#include "hccl/hccl_res.h"
#include "../../../../../../legacy/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/unified_platform/resource/buffer/local_ipc_rma_buffer.h"
#include "../../../../../../legacy/unified_platform/resource/buffer/remote_rma_buffer.h"
#include "../../../../../../legacy/common/binary_stream.h"
#include "buffer.h"
#include "transport_status.h"

namespace hcomm {
class AivUbMemTransport{
public:
    MAKE_ENUM(AivUbMemTransportStatus, INIT, SOCKET_OK, SEND_DATA_SIZE, RECV_DATA_SIZE, SEND_MEM_INFO, RECV_MEM_INFO,
        RECV_MEM_FIN, CONNECT_FAILED, SOCKET_TIMEOUT, READY);
    AivUbMemTransport(Hccl::Socket *socket, HcommChannelDesc &channelDesc);
    ~AivUbMemTransport() = default;
    HcclResult FillTagVec(HcommMemHandle *memHandles, uint32_t bufferNum, std::vector<Hccl::LocalIpcRmaBuffer *> &bufferVec,
        std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> &tagVec);
    HcclResult Init();
    Hccl::TransportStatus GetStatus();
    HcclResult GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags);
    HcclResult GetMemTag(char **memTag, uint32_t memNum);
    HcclResult GetUserRemoteMem(CommMem **remoteMem, char ***memTags, uint32_t *memNum);
    HcclResult CheckSocketStatus(std::string socketOpreator);
    HcclResult UpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum);

private:
    Hccl::Socket *socket_{}; // 交换所用的socket
    HcommChannelDesc channelDesc_;
    uint32_t exchangeDataSize_{0};
    std::vector<HcclMem> remoteMems_;
    std::vector<CommMem> remoteUserMems_;
    std::vector<std::string> tagCopies_; //储存memTag字符串副本
    std::vector<char*> tagPointers_; // 储存指针
    bool cacheValid_ = false; // GetUserRemoteMem 的缓存标识
    
    std::vector<Hccl::LocalIpcRmaBuffer *>  localRmaBufferVec_{};
    std::vector<Hccl::LocalIpcRmaBuffer *>  locMemTemp_{};
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> localUserMemTag_{}; 
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> locTagTemp_{}; 
    std::vector<std::unique_ptr<Hccl::RemoteIpcRmaBuffer>> rmtBufferVec_{};
    std::vector<Hccl::RemoteRmaBuffer *> rmtRmaBufferVec_{};
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> remoteUserMemTag_{};
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> rmtTagTemp_{};
    AivUbMemTransportStatus aivUbStatus_{AivUbMemTransportStatus::INVALID};
    Hccl::TransportStatus baseStatus_{Hccl::TransportStatus::INVALID};
    std::mutex remoteMemsMutex_;     // 远端内存列表互斥锁

    std::vector<char> sendData_{};
    std::vector<char> recvData_{};
    
    HcclResult IsSocketReady(bool &isReady);
    HcclResult SendDataSize();
    HcclResult RecvDataSize();
    HcclResult SendMemInfo();
    HcclResult RecvMemInfo();
    HcclResult RecvDataProcess();
    void BufferPack(Hccl::BinaryStream &binaryStream, std::vector<Hccl::LocalIpcRmaBuffer *> &bufferVec,
        std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> &localUserMemTag) const;
    void RmtBufferUnpackProc(Hccl::BinaryStream &binaryStream);
    HcclResult StateMachine();
    Hccl::TransportStatus UpdateStatus();
    void CheckStatusFuncResult(std::string funcName, HcclResult ret);
};

}  // namespace hcomm

#endif  // AIV_UB_MEM_TRANSPORT_H
