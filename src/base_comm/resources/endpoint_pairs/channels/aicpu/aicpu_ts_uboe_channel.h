/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_TS_UBOE_CHANNEL_H
#define AICPU_TS_UBOE_CHANNEL_H

#include "channel.h"
#include "socket_mgr.h"

// Orion
#include "../../../../../../src/legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../../src/legacy/ascend950/framework/resource_manager/socket/socket_manager.h"
#include "buffer_key.h"
#include "rma_connection.h"
#include "ub_mem_transport.h"
#include "dev_ub_connection.h"
#include "ub_local_notify.h"
#include "aicpu_ts_channel_helper.h"

namespace hcomm {

constexpr u32    FINISH_MSG_SIZE             = 128;
constexpr char_t FINISH_MSG[FINISH_MSG_SIZE] = "Uboe Comm Pipe ready!";

class AicpuTsUboeChannel : public Channel {
public:
    AicpuTsUboeChannel(EndpointHandle endpointHandle, const HcommChannelDesc &channelDesc);
    ~AicpuTsUboeChannel() override;

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t *notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos) override;
    ChannelStatus GetStatus() override;

    HcclResult UpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum) override;

    HcclResult H2DResPack(std::vector<char>& buffer);

    HcclResult Clean() override;
    HcclResult Resume() override;
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }
    
    HcommChannelKind GetChannelKind() const override
    {
        return HcommChannelKind::AICPU_TS_UBOE;
    }

    // 数据面接口
    HcclResult WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult Write(void *dst, const void *src, uint64_t len) override;
    HcclResult Read(void *dst, const void *src, uint64_t len) override;
    HcclResult ChannelFence() override;

    AicpuTsChannelHelper *GetAicpuTsHelper() override { return &aicpuTsHelper_; }

private:
    MAKE_ENUM(UboeRmtBufType, NOTIFY, BUFFER)
    using RemoteBufferVec = std::vector<std::unique_ptr<Hccl::RemoteUbRmaBuffer>>;
    using LocalBufferVec = std::vector<Hccl::LocalUbRmaBuffer *>;

    HcclResult Makebufs(HcommMemHandle *memHandles, uint32_t memHandleNum,
        std::vector<std::shared_ptr<Hccl::Buffer>> &bufs);
    HcclResult ParseInputParam();
    HcclResult BuildConnection();
    HcclResult BuildNotify();
    HcclResult BuildBuffer(std::vector<std::shared_ptr<Hccl::Buffer>> &bufs);
    HcclResult BuildUbMemTransport();
    HcclResult BuildSocket();

    std::vector<char> GetUniqueIdV2();
    HcclResult PackOpData(std::vector<char> &data);

    HcclResult CheckSocketStatus(const std::string &socketOperator);
    bool IsSocketReady();
    bool IsResReady();
    bool IsConnsReady();
    bool RecvDataProcess();
    void RecvEidDataProcess();
    void SendDataSize();
    void RecvDataSize();
    void SendEidData();
    void SendExchangeData();
    void RecvExchangeData();
    void RecvEidData();
    void SendFinish();
    void RecvFinish();

    void HandleProcessData();
    void ProcessUboeState();

    void EidPack();
    void NotifyVecPack(Hccl::BinaryStream &binaryStream);
    void BufferVecPack(Hccl::BinaryStream &binaryStream, std::vector<Hccl::LocalRmaBuffer *> &bufferVec);
    void ConnVecPack(Hccl::BinaryStream &binaryStream);
    void RmtEidUnpackProc(Hccl::IpAddress& rmtAddr);
    void RmtBufferVecUnpackProc(u32 locNum, Hccl::BinaryStream &binaryStream,
        RemoteBufferVec &bufferVec, UboeRmtBufType type);
    bool ConnVecUnpackProc(Hccl::BinaryStream &binaryStream);
    void BuildConn();

    std::vector<char> GetNotifyUniqueIds();
    std::vector<char> GetRmtBufferUniqueIds(RemoteBufferVec &bufferVec, UboeRmtBufType type) const;
    std::vector<char> GetLocBufferUniqueIds(LocalBufferVec &bufferVec, UboeRmtBufType type) const;
    std::vector<char> GetSingleRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId) const;
    std::vector<char> GetSingleLocBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue) const;
    std::vector<char> GetConnUniqueIds();
private:
    // --------------------- 入参 ---------------------
    AicpuTsChannelHelper                                        aicpuTsHelper_;
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
    Hccl::BaseMemTransport::CommonLocRes                        commonRes_{};
    std::vector<Hccl::LocalRmaBuffer *>                         bufferVecTemp_; // channel 复用时暂存新增 rmaBuffer
    std::vector<std::unique_ptr<Hccl::DevUbConnection>>         connections_{};
    std::vector<std::unique_ptr<Hccl::LocalUbRmaBuffer>>        localRmaBuffers_{};
    std::vector<std::unique_ptr<Hccl::UbLocalNotify>>           localNotifies_{};
    std::unique_ptr<Hccl::Socket>                               serverSocket_;

    ChannelStatus                                               channelStatus{ChannelStatus::INIT};
    MAKE_ENUM(UboeStatus, INIT, SEND_EID, RECV_EID, PROCESS_EID_DATA, BUILD_CONN, SEND_SIZE, RECV_SIZE, SEND_DATA, 
        RECV_DATA, SEND_FIN, RECV_FIN, PROCESS_DATA, SET_READY, READY)
    UboeStatus uboeStatus{UboeStatus::INIT};

    u32 bufferNum_{0};
    u32 connNum_{0};
    u32 recvDataSize_{0};

    RemoteBufferVec rmtNotifyVec_;     // 远端 notify
    RemoteBufferVec rmtBufferVec_;     // 远端 buffer
    LocalBufferVec  locBufferVec_;     // 本端 buffer
    std::vector<char> recvData_{};
    std::vector<char> recvFinishMsg_{};
    std::vector<char> recvEidData_{};
    std::vector<char> sendData_{};
    std::vector<char> sendFinishMsg_{};
    std::vector<char> sendEidData_{};
    bool isRecvFirst_{false};
    
    Hccl::IpAddress     locAddr_;
    Hccl::IpAddress     rmtAddr_;

    std::mutex remoteMemsMutex_; // 远端内存列表互斥锁
    bool cacheValid_ = false; // 当前缓存是否有效
    std::vector<CommMem>         remoteUserMems_;     // 内存基本信息缓存
    std::vector<std::string>     memInfoCopies_;          // 储存 Tag 字符串副本
    std::vector<char*>           memInfoPointers_;        // Tag 缓存
    const Hccl::SocketConfig*    socketConfig_{nullptr};
    uint32_t    devicePhyId_{};
};

} // namespace hcomm

#endif // AICPU_TS_UBOE_CHANNEL_H