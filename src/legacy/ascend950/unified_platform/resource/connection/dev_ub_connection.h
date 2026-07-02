/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_DEV_UB_CONNECTION_H
#define HCCLV2_DEV_UB_CONNECTION_H

#include "rma_connection.h"
#include "op_mode.h"
#include "orion_adapter_hccp.h"
#include "../../../framework/env_config/env_config.h"
#include "tp_manager.h"
#include "local_ub_rma_buffer.h"
#include "stream.h"
#include "task.h"
#include "mc2_type.h"
#include "hcomm/hcomm_res_entity_defs.h"

namespace Hccl {

class DevUbConnection : public RmaConnection {
public:
    DevUbConnection(const RdmaHandle rdmaHandle, const IpAddress &locAddr, const IpAddress &rmtAddr,
                    const OpMode opMode, const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
                    const IpAddress &locIpv4Addr = IpAddress(), const IpAddress &rmtIpv4Addr = IpAddress(),
                    u8 qos = static_cast<u8>(UB_QOS_DEFAULT));
    void          Connect() override;
    RmaConnStatus GetStatus() override;
    bool          Suspend() override;

    std::unique_ptr<Serializable> GetExchangeDto() override;
    void                          ParseRmtExchangeDto(const Serializable &rmtDto) override;
    void                          ImportRmtDto() override;

    std::vector<char> GetUniqueId() const override;

    void SetCqInfo(HcclAiRMACQ &cq) const;
 	 
 	void SetWqInfo(HcclAiRMAWQ &wq) const;
    
    void SetCqContextInfo(CqContext &cq) const;
    void SetSqContextInfo(SqContext &sq) const;

    unique_ptr<BaseTask> PrepareRead(const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
                                     const SqeConfig &config) override;

    unique_ptr<BaseTask> PrepareReadReduce(const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
                                           DataType dataType, ReduceOp reduceOp, const SqeConfig &config) override;

    unique_ptr<BaseTask> PrepareWrite(const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
                                      const SqeConfig &config) override;

    unique_ptr<BaseTask> PrepareWriteReduce(const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
                                            DataType dataType, ReduceOp reduceOp, const SqeConfig &config) override;

    unique_ptr<BaseTask> PrepareInlineWrite(const MemoryBuffer &remoteMemBuf, u64 data,
                                            const SqeConfig &config) override;

    unique_ptr<BaseTask> PrepareWriteWithNotify(const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
                                                u64 data, const MemoryBuffer &remoteNotifyMemBuf,
                                                const SqeConfig &config) override;

    unique_ptr<BaseTask> PrepareWriteReduceWithNotify(const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
                                                      DataType dataType, ReduceOp reduceOp, u64 data,
                                                      const MemoryBuffer &remoteNotifyMemBuf,
                                                      const SqeConfig    &config) override;

    class UbCiUpdater;

    void AddNop(const Stream &stream) override;

    void         ReleaseTp();
    ~DevUbConnection() override;

    string Describe() const override;
    HcclResult Describe(std::string &dfxMsg) override;

    HrtUbJfcMode GetUbJfcMode() const;
    JettyHandle& GetJettyHandle();
    JettyHandle& GetRemoteJettyHandle();
    RdmaHandle&  GetRdmaHandle();
    u32          GetPiVal() const;
    u32          GetCiVal() const;
    u32          GetSqDepth() const;

protected:
    TpProtocol     tpProtocol{TpProtocol::INVALID};
    void           GetTimeOut();
    u8             jettyTimeOut{8};

private:
    MAKE_ENUM(UbConnStatus,
        INIT, TP_INFO_GETTING, JETTY_CREATING, JETTY_CREATED,
        JETTY_IMPORTING,
        READY,
        CONN_INVALID);

    UbConnStatus ubConnStatus{UbConnStatus::INIT};

    RdmaHandle   rdmaHandle{nullptr};
    IpAddress    locAddr{};
    IpAddress    rmtAddr{};
    OpMode       opMode{OpMode::OPBASE};
    HrtUbJfcMode jfcMode{HrtUbJfcMode::STARS_POLL};
    IpAddress    locIpv4Addr{};
    IpAddress    rmtIpv4Addr{};
    u32          tokenValue{GetUbToken()};
    Eid          rmtEid{};
    Eid          locEid{};
    u8           qos_{static_cast<u8>(UB_QOS_DEFAULT)}; // 业务 QoS，GetTpInfo / ReleaseTpInfo 缓存键

    bool         devUsed_{false};

    int32_t   devLogicId{0};
    u32       dieId{0};
    u32       funcId{0};
    JfcHandle jfcHandle{0};
    u32       sqDepth{0};
    uint64_t  sqBuffVa{0};

    RequestHandle  reqHandle{0};
    vector<char_t> reqDataBuffer;

    u8             remoteQpKey[HRT_UB_QP_KEY_MAX_LEN] = {0};
    u32            keySize{0};
    u32            remoteTokenValue{0};
    JettyImportCfg jettyImportCfg{};
    void          *remoteJettyHandlePtr{nullptr};

    JettyHandle jettyHandle{0};
    void       *jettyHandlePtr{nullptr};
    JettyHandle remoteJettyHandle{0};
    u8          localQpKey[HRT_UB_QP_KEY_MAX_LEN]{0};

    u32 jettyId{0};
    u64 dbAddr{0};
    u32 tpn{0};

    u32                 localTpnStart{0};
    u32                 localTpNum{0};
    TpInfo              tpInfo{};

    u32 piVal{0};
    u32 ciVal{0};

    CqCreateInfo cqInfo_{0};

    bool isdevUsed{false};

    bool CheckRequestResult();
    void ThrowAbnormalStatus(std::string funcName);
    void AdvanceUbConnFromInit();
    void AdvanceUbConnFromTpInfoGetting();
    void AdvanceUbConnAfterTpInfoReady();
    void AdvanceUbConnFromJettyCreating();
    void AdvanceUbConnFromJettyCreated();
    void AdvanceUbConnFromJettyImporting();

    void         GenerateLocalPsn();
    void         CreateJetty(const bool devUsed);
    void         SetJettyInfo();
    bool         GetTpInfo();
    void         UpdateLocTpInfo();
    TpInfo       SelectTpInfo();
    void         ImportJetty();
    void         SetImportInfo();
    void         UnImportJetty();
    void         DestroyJetty();
    void         ReleaseResource();

    void ProcessSlices(const MemoryBuffer &loc, const MemoryBuffer &rmt,
                       std::function<void(const MemoryBuffer &, const MemoryBuffer &, u32)> processOneSlice,
                       DataType dataType = DataType::INVALID) const;

    void ProcessSlicesWithNotify(const MemoryBuffer &loc, const MemoryBuffer &rmt,
                                 std::function<void(const MemoryBuffer &, const MemoryBuffer &, u32)> processOneSlice,
                                 std::function<void(const MemoryBuffer &, const MemoryBuffer &)> processOneSliceWithNotify,
                                 DataType dataType = DataType::INVALID) const;
    
    std::unique_ptr<BaseTask> ConstructTaskUbSend(const HrtRaUbSendWrRespParam &sendWrResp, const SqeConfig &config);
    void                      UpdateCiVal(u32 ci);
    HcclResult                CalcTotalTimeout(uint32_t &outTotalTimeoutMs);
};

class DevUbTpConnection : public DevUbConnection {
public:
    DevUbTpConnection(const RdmaHandle rdmaHandle, const IpAddress &locAddr, const IpAddress &rmtAddr,
                      const OpMode opMode, const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
                      const IpAddress &locIpv4Addr = IpAddress(), const IpAddress &rmtIpv4Addr = IpAddress(),
                      u8 qos = static_cast<u8>(UB_QOS_DEFAULT));
};

class DevUbCtpConnection : public DevUbConnection {
public:
    DevUbCtpConnection(const RdmaHandle rdmaHandle, const IpAddress &locAddr, const IpAddress &rmtAddr,
                       const OpMode opMode, const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
                       const IpAddress &locIpv4Addr = IpAddress(), const IpAddress &rmtIpv4Addr = IpAddress(),
                       u8 qos = static_cast<u8>(UB_QOS_DEFAULT));
};

class DevUbUboeConnection : public DevUbConnection {
public:
    DevUbUboeConnection(const RdmaHandle rdmaHandle, const IpAddress &locAddr, const IpAddress &rmtAddr,
                        const OpMode opMode, const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
                        const IpAddress &locIpv4Addr = IpAddress(), const IpAddress &rmtIpv4Addr = IpAddress(),
                        u8 qos = static_cast<u8>(UB_QOS_DEFAULT));
};

std::vector<DevUbConnection *> GetStarsPollUbConns(const std::vector<RmaConnection *> &rmaConns);

bool IfNeedUpdatingUbCi(const std::vector<DevUbConnection *> &ubConns);

} // namespace Hccl

#endif // HCCLV2_DEV_UB_CONNECTION_H
