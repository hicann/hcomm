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
#include <functional>

namespace Hccl {

class DevUbConnection : public RmaConnection {
public:
    DevUbConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locIpv4Addr = IpAddress(), const IpAddress& rmtIpv4Addr = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), CommEngine engine = COMM_ENGINE_RESERVED,
        u32 sqDepth = UB_SQ_DEPTH_NOT_SET);
    void Connect() override;
    RmaConnStatus GetStatus() override;
    bool Suspend() override;

    std::unique_ptr<Serializable> GetExchangeDto() override;
    void ParseRmtExchangeDto(const Serializable& rmtDto) override;
    void ImportRmtDto() override;

    std::vector<char> GetUniqueId() const override;

    void SetCqInfo(HcclAiRMACQ& cq) const;

    void SetWqInfo(HcclAiRMAWQ& wq) const;

    void SetCqContextInfo(CqContext& cq) const;
    void SetSqContextInfo(SqContext& sq) const;

    unique_ptr<BaseTask>
    PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareReadReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
        const SqeConfig& config) override;

    unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
        const SqeConfig& config) override;

    unique_ptr<BaseTask>
    PrepareInlineWrite(const MemoryBuffer& remoteMemBuf, u64 data, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareWriteWithNotify(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, u64 data,
        const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareWriteReduceWithNotify(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
        u64 data, const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config) override;

    class UbCiUpdater;

    void AddNop(const Stream& stream) override;

    /**
     * @brief 注入共享 jetty 模式：复用外部已创建的 jetty 句柄，connection 不再自建/自销毁 jetty。
     *        必须在 connection 构造后、Connect/GetStatus 推进状态机前调用。
     *        调用后状态机跳过 JETTY_CREATING，直接进入 JETTY_CREATED。
     * @note 架构说明：本组方法属 base_comm 共享 jetty 特性（IS_SHARED_QUEUE）的实现细节，
     *       因 DevUbConnection 当前仍位于 legacy/ 而暂置于此。base_comm 侧通过
     *       shared_jetty_connection_adapter 适配层调用，不直接依赖本类。
     * @param[in] jettyHdl 共享 jetty 句柄
     * @param[in] jettyHdlPtr 底层 jetty 指针（用于 HrtRaUbPostSend 等）
     * @param[in] jId jetty id
     * @param[in] sqVa SQ 缓冲 VA
     * @param[in] db doorbell 地址
     * @param[in] qpKey 本地 QP key
     * @param[in] kSize key 长度
     * @param[in] sDepth SQ 深度
     * @param[in] tpHdl 创建共享 jetty 时使用的 TP handle（注入后主 connection 复用此 tpHandle，
     *                  避免重新向管控面申请得到不同 tpHandle 导致对端 import 路由不匹配）
     * @param[in] epTag Endpoint 不透明标签（透传给 releaseCb 供回调定位 Endpoint）
     * @param[in] releaseCb connection 销毁时调用的释放回调（由 base_comm 层注入 Endpoint::ReleaseSharedJetty）
     */
    HcclResult InjectSharedJetty(
        JettyHandle jettyHdl, void* jettyHdlPtr, uint32_t jId, uint64_t sqVa, uint64_t db, const uint8_t* qpKey,
        uint32_t kSize, uint32_t sDepth, uint64_t tpHdl, void* epTag, std::function<void(void*)> releaseCb);

    /**
     * @brief 将已自建 jetty 的 connection 标记为共享所有权移交：之后析构不再销毁 jetty，
     *        jetty 生命周期交由 Endpoint::sharedJettyCtx_ 管理。仅当 connection 已完成 SetJettyInfo 后调用。
     */
    void TransferJettyOwnership();

    /**
     * @brief jetty 衍生字段集合，供共享模式下提取注入给其他 connection
     */
    struct JettyInfo {
        JettyHandle handle{0};
        void* handlePtr{nullptr};
        uint32_t jettyId{0};
        uint64_t sqBuffVa{0};
        uint64_t dbAddr{0};
        uint8_t localQpKey[HRT_UB_QP_KEY_MAX_LEN]{0};
        uint32_t keySize{0};
        uint32_t sqDepth{0};
        uint64_t tpHandle{0};
        RdmaHandle rdmaHandle{nullptr}; // 销毁 JFC 所需的 RDMA 句柄
        JfcHandle jfcHandle{0};         // 临时 connection 创建的 JFC, 由 Endpoint 统一销毁
    };

    /** 获取当前 connection 的 jetty 衍生字段（共享模式下用于 Adopt 到 Holder） */
    HcclResult GetJettyInfo(JettyInfo& info) const;

    void ReleaseTp();
    ~DevUbConnection() override;

    string Describe() const override;
    HcclResult Describe(std::string& dfxMsg) override;

    HrtUbJfcMode GetUbJfcMode() const;
    JettyHandle& GetJettyHandle();
    JettyHandle& GetRemoteJettyHandle();
    RdmaHandle& GetRdmaHandle();
    u32 GetPiVal() const;
    u32 GetCiVal() const;
    u32 GetSqDepth() const;

    void SetMaxReadSize(u32 value);
    void SetMaxWriteSize(u32 value);

protected:
    TpProtocol tpProtocol{TpProtocol::INVALID};
    void GetTimeOut();
    u8 jettyTimeOut{8};

private:
    MAKE_ENUM(UbConnStatus, INIT, TP_INFO_GETTING, JETTY_CREATING, JETTY_CREATED, JETTY_IMPORTING, READY, CONN_INVALID);

    UbConnStatus ubConnStatus{UbConnStatus::INIT};

    RdmaHandle rdmaHandle{nullptr};
    IpAddress locAddr{};
    IpAddress rmtAddr{};
    OpMode opMode{OpMode::OPBASE};
    HrtUbJfcMode jfcMode{HrtUbJfcMode::STARS_POLL};
    CommEngine engine_{COMM_ENGINE_RESERVED};
    IpAddress locIpv4Addr{};
    IpAddress rmtIpv4Addr{};
    u32 tokenValue{GetUbToken()};
    Eid rmtEid{};
    Eid locEid{};
    u8 qos_{static_cast<u8>(UB_QOS_DEFAULT)}; // 业务 QoS，GetTpInfo / ReleaseTpInfo 缓存键

    bool devUsed_{false};

    int32_t devLogicId{0};
    u32 dieId{0};
    u32 funcId{0};
    JfcHandle jfcHandle{0};
    u32 sqDepth{0};
    uint64_t sqBuffVa{0};

    RequestHandle reqHandle{0};
    vector<char_t> reqDataBuffer;

    u8 remoteQpKey[HRT_UB_QP_KEY_MAX_LEN] = {0};
    u32 keySize{0};
    u32 remoteTokenValue{0};
    JettyImportCfg jettyImportCfg{};
    void* remoteJettyHandlePtr{nullptr};

    JettyHandle jettyHandle{0};
    void* jettyHandlePtr{nullptr};
    JettyHandle remoteJettyHandle{0};
    u8 localQpKey[HRT_UB_QP_KEY_MAX_LEN]{0};

    u32 jettyId{0};
    u64 dbAddr{0};
    u32 tpn{0};

    u32 localTpnStart{0};
    u32 localTpNum{0};
    TpInfo tpInfo{};

    u32 piVal{0};
    u32 ciVal{0};

    CqCreateInfo cqInfo_{0};

    bool isdevUsed{false};

    // 最大传输size，切片使用
    u32 maxReadSize{0};
    u32 maxWriteSize{0};

    // 共享 jetty 注入模式标记：true 表示复用外部 jetty，不自建/自销毁
    bool isSharedJetty_{false};
    void* endpointTag_{nullptr};                    // 共享模式下透传给 releaseCb_ 的 Endpoint 标签
    std::function<void(void*)> releaseCb_{nullptr}; // 共享 jetty 释放回调（调 Endpoint::ReleaseSharedJetty）

    bool CheckRequestResult();
    void ThrowAbnormalStatus(std::string funcName);
    void AdvanceUbConnFromInit();
    void AdvanceUbConnFromTpInfoGetting();
    void AdvanceUbConnAfterTpInfoReady();
    void AdvanceUbConnFromJettyCreating();
    void AdvanceUbConnFromJettyCreated();
    void AdvanceUbConnFromJettyImporting();

    void GenerateLocalPsn();
    void CreateJetty(const bool devUsed);
    void CreateAivUrmaJfc();
    void SetJettyInfo();
    bool GetTpInfo();
    void UpdateLocTpInfo();
    TpInfo SelectTpInfo();
    void ImportJetty();
    void SetImportInfo();
    void UnImportJetty();
    void DestroyJetty();
    void ReleaseResource();

    void ProcessSlices(
        const MemoryBuffer& loc, const MemoryBuffer& rmt,
        std::function<void(const MemoryBuffer&, const MemoryBuffer&, u32)> processOneSlice,
        DataType dataType = DataType::INVALID) const;

    void ProcessSlicesWithNotify(
        const MemoryBuffer& loc, const MemoryBuffer& rmt,
        std::function<void(const MemoryBuffer&, const MemoryBuffer&, u32)> processOneSlice,
        std::function<void(const MemoryBuffer&, const MemoryBuffer&)> processOneSliceWithNotify,
        DataType dataType = DataType::INVALID) const;

    std::unique_ptr<BaseTask> ConstructTaskUbSend(const HrtRaUbSendWrRespParam& sendWrResp, const SqeConfig& config);
    void UpdateCiVal(u32 ci);
    HcclResult CalcTotalTimeout(uint32_t& outTotalTimeoutMs);
};

class DevUbTpConnection : public DevUbConnection {
public:
    DevUbTpConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locIpv4Addr = IpAddress(), const IpAddress& rmtIpv4Addr = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), CommEngine engine = COMM_ENGINE_RESERVED,
        u32 sqDepth = UB_SQ_DEPTH_NOT_SET);
};

class DevUbCtpConnection : public DevUbConnection {
public:
    DevUbCtpConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locIpv4Addr = IpAddress(), const IpAddress& rmtIpv4Addr = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), CommEngine engine = COMM_ENGINE_RESERVED,
        u32 sqDepth = UB_SQ_DEPTH_NOT_SET);
};

class DevUbUboeConnection : public DevUbConnection {
public:
    DevUbUboeConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locIpv4Addr = IpAddress(), const IpAddress& rmtIpv4Addr = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), CommEngine engine = COMM_ENGINE_RESERVED);
};

class DevUbUbgConnection : public DevUbConnection {
public:
    DevUbUbgConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locAddrEid = IpAddress(), const IpAddress& rmtAddrEid = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), CommEngine engine = COMM_ENGINE_RESERVED,
        u32 sqDepth = UB_SQ_DEPTH_NOT_SET);
};

std::vector<DevUbConnection*> GetStarsPollUbConns(const std::vector<RmaConnection*>& rmaConns);

bool IfNeedUpdatingUbCi(const std::vector<DevUbConnection*>& ubConns);

} // namespace Hccl

#endif // HCCLV2_DEV_UB_CONNECTION_H
