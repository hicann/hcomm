/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_transport_.h"

#include "exception_handler.h"

#include "../../../../../legacy/unified_platform/resource/mem/user_remote_mem_getter.h"

#include "env_config/env_config.h"

namespace hcomm {

constexpr uint32_t FINISH_MSG_SIZE = 128;
constexpr char FINISH_MSG[FINISH_MSG_SIZE] = "Transport exchange data ready!";

HcclResult BuildCcuConnection(const CcuTransport::CcuConnectionInfo &ccuConnectionInfo, 
    std::unique_ptr<CcuConnection> &ccuConnection)
{
    if (ccuConnectionInfo.type == CcuTransport::CcuConnectionType::UBC_CTP) {
        ccuConnection.reset(new (std::nothrow) CcuCtpConnection(
            ccuConnectionInfo.locAddr, ccuConnectionInfo.rmtAddr,
            ccuConnectionInfo.channelInfo, ccuConnectionInfo.ccuJettys, ccuConnectionInfo.qos));
    } else {
        ccuConnection.reset(new (std::nothrow) CcuRtpConnection(
            ccuConnectionInfo.locAddr, ccuConnectionInfo.rmtAddr,
            ccuConnectionInfo.channelInfo, ccuConnectionInfo.ccuJettys, ccuConnectionInfo.qos));
    }
    CHK_PTR_NULL(ccuConnection);
    CHK_RET(ccuConnection->Init());
    return HCCL_SUCCESS;
}

HcclResult CcuCreateTransport(Hccl::Socket *socket, const CcuTransport::CcuConnectionInfo &ccuConnectionInfo,
    const CcuTransport::CclBufferInfo &cclBufferInfo, std::unique_ptr<CcuTransport> &ccuTransport)
{
    CHK_PTR_NULL(socket);
    std::unique_ptr<CcuConnection> ccuConnection{nullptr};
    CHK_RET(BuildCcuConnection(ccuConnectionInfo, ccuConnection));

    ccuTransport.reset(new (std::nothrow)
        CcuTransport(socket, std::move(ccuConnection), cclBufferInfo));
    CHK_PTR_NULL(ccuTransport);
    CHK_RET(ccuTransport->Init());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuCreateTransport(Hccl::Socket *socket, const CcuTransport::CcuConnectionInfo &ccuConnectionInfo,
    const std::vector<CcuTransport::CclBufferInfo> &bufferInfos, std::unique_ptr<CcuTransport> &ccuTransport)
{
    CHK_PTR_NULL(socket);
    std::unique_ptr<CcuConnection> ccuConnection{nullptr};
    CHK_RET(BuildCcuConnection(ccuConnectionInfo, ccuConnection));

    if (bufferInfos.size() == 0) {
        HCCL_ERROR("[CcuCreateTransport] bufferNum is 0.");
        return HCCL_E_PARA;
    }
    ccuTransport.reset(new (std::nothrow)
        CcuTransport(socket, std::move(ccuConnection), bufferInfos));
    CHK_PTR_NULL(ccuTransport);
    // 可能申请xn cke失败，需要回退
    auto ret = ccuTransport->Init();
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING("[%s] ccuTransport init failed, ccu transport resources unavailable.", __func__);
        return ret;
    }
    CHK_RET(ret);

    return HcclResult::HCCL_SUCCESS;
}

CcuTransport::CcuTransport(Hccl::Socket *socket, std::unique_ptr<CcuConnection> &&connection,
    const CclBufferInfo &locCclBufInfo)
    : socket_(socket), ccuConnection_(std::move(connection))
{
    locBufferInfos_.push_back(locCclBufInfo);
}

CcuTransport::CcuTransport(Hccl::Socket *socket, std::unique_ptr<CcuConnection> &&connection,
    const std::vector<CclBufferInfo> &bufferInfos)
    : socket_(socket), ccuConnection_(std::move(connection)), locBufferInfos_(bufferInfos)
{
}

HcclResult CcuTransport::Init()
{
    dieId_      = ccuConnection_->GetDieId();
    devLogicId_ = ccuConnection_->GetDevLogicId();
    auto ret = AppendCkes(INIT_CKE_NUM);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        return ret;
    }
    CHK_RET(ret);

    ret = AppendXns(INIT_XN_NUM);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        return ret;
    }
    CHK_RET(ret);

    transStatus_ = TransStatus::INIT;
    return HCCL_SUCCESS;
}

CcuTransport::TransStatus CcuTransport::GetStatus()
{
    if (transStatus_ == TransStatus::READY
        || transStatus_ == TransStatus::CONNECT_FAILED
        || transStatus_ == TransStatus::SOCKET_TIMEOUT) {
        return transStatus_;
    }

    if (StatusMachine() != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuTransport][%s] failed, %s.",
            __func__, transStatus_.Describe().c_str());
        transStatus_ = TransStatus::CONNECT_FAILED;
    }

    return transStatus_;
}

HcclResult CcuTransport::AppendCkes(uint32_t ckesNum)
{
    std::vector<ResInfo> resInfo;
    auto ret = CcuDevMgrImp::AllocCke(devLogicId_, dieId_, ckesNum, resInfo);
    CHK_PRT_RET(ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING("[CcuTransport][%s] failed, the resource is not enough.",
            __func__),
        ret);
    CHK_RET(ret);

    const uint32_t resSize = resInfo.size();
    for (uint32_t i = 0; i < resSize; i++) {
        const uint32_t ckeNum     = resInfo[i].num;
        const uint32_t ckesSartId = resInfo[i].startId;
        for (uint32_t j = 0; j < ckeNum; j++) {
            locRes_.ckes.emplace_back(ckesSartId + j);
        }
    }
    ckesRes_.push_back(resInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::AppendXns(uint32_t xnsNum)
{
    std::vector<ResInfo> resInfo;
    auto ret = CcuDevMgrImp::AllocXn(devLogicId_, dieId_, xnsNum, resInfo);
    CHK_PRT_RET(ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING("[CcuTransport][%s] failed, the resource is not enough.",
            __func__),
        ret);
    CHK_RET(ret);

    const uint32_t resSize = resInfo.size();
    for (uint32_t i = 0; i < resSize; i++) {
        uint32_t xnNum     = resInfo[i].num;
        uint32_t xnsSartId = resInfo[i].startId;
        for (uint32_t j = 0; j < xnNum; j++) {
            locRes_.xns.emplace_back(xnsSartId + j);
        }
    }
    xnsRes_.push_back(resInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::StatusMachine()
{
    EXCEPTION_HANDLE_BEGIN
    Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
    if (socketStatus == Hccl::SocketStatus::INIT || socketStatus == Hccl::SocketStatus::TIMEOUT) {
        HCCL_ERROR("[CcuTransport][GetStatus]socket timeout or no link, please check");
        return HcclResult::HCCL_E_INTERNAL;
    }
    
    if (socketStatus != Hccl::SocketStatus::OK) {
        return HcclResult::HCCL_SUCCESS; // 操作成功，保持当前状态
    }
    EXCEPTION_HANDLE_END

    switch (transStatus_) {
        case CcuTransport::TransStatus::INIT: {
            auto connStatus = ccuConnection_->GetStatus();
            if (connStatus == CcuConnStatus::CONN_INVALID) {
                HCCL_ERROR("[CcuTransport][GetStatus] connection status[%s] failed."
                    " please check.", connStatus.Describe().c_str());
                return HcclResult::HCCL_E_INTERNAL;
            }

            if (connStatus == CcuConnStatus::EXCHANGEABLE
                || connStatus == CcuConnStatus::CONNECTED) {
                // connection完成本端资源创建或复用时，发送本端资源信息
                CHK_RET(SendDataSize());
                transStatus_ = TransStatus::SEND_DATA_SIZE;
            }

            // connection状态非错误但未达到目标状态时，transport保持当前状态
            break;
        }
        case CcuTransport::TransStatus::SEND_DATA_SIZE:
            CHK_RET(RecvDataSize());
            transStatus_ = TransStatus::RECV_DATA_SIZE;
            break;
        case CcuTransport::TransStatus::RECV_DATA_SIZE:
            CHK_RET(SendConnAndTransInfo());
            transStatus_ = TransStatus::SEND_ALL_INFO;
            break;
        case CcuTransport::TransStatus::SEND_ALL_INFO:
            CHK_RET(RecvConnAndTransInfo());
            transStatus_ = TransStatus::RECV_ALL_INFO;
            break;
        case CcuTransport::TransStatus::RECV_ALL_INFO:
            CHK_RET(RecvDataProcess());
            CHK_RET(ccuConnection_->ImportJetty());
            transStatus_ = TransStatus::SEND_FIN;
            break;
        case CcuTransport::TransStatus::SEND_FIN: {
            auto connStatus = ccuConnection_->GetStatus();
            if (connStatus == CcuConnStatus::CONN_INVALID) {
                HCCL_ERROR("[CcuTransport][GetStatus] connection status[%s] failed."
                    " please check", connStatus.Describe().c_str());
                return HcclResult::HCCL_E_INTERNAL;
            }

            if (connStatus == CcuConnStatus::CONNECTED) {
                CHK_RET(SendFinish());
                transStatus_ = CcuTransport::TransStatus::RECVING_FIN;
            }
            break;
        }
        case CcuTransport::TransStatus::RECVING_FIN:
            CHK_RET(RecvFinish());
            transStatus_ = CcuTransport::TransStatus::RECV_FIN;
            break;
        case CcuTransport::TransStatus::RECV_FIN:
            CHK_RET(CheckFinish());
            transStatus_ = CcuTransport::TransStatus::READY;
            break;
        case CcuTransport::TransStatus::SEND_TRANS_RES:
            CHK_RET(SendTransInfo());
            transStatus_ = CcuTransport::TransStatus::RECVING_TRANS_RES;
            break;
        case CcuTransport::TransStatus::RECVING_TRANS_RES:
            CHK_RET(RecvTransInfo());
            transStatus_ = CcuTransport::TransStatus::RECV_TRANS_RES;
            break;
        case CcuTransport::TransStatus::RECV_TRANS_RES:
            CHK_RET(RecvTransInfoProcess());
            transStatus_ = CcuTransport::TransStatus::SEND_FIN;
            break;
        default:
            HCCL_ERROR("[CcuTransport][%s] failed, error status[%s].",
                __func__, transStatus_.Describe().c_str());
            transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
            break;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::SendDataSize()
{
    Hccl::BinaryStream binaryStream;
    CHK_RET(HandshakeMsgPack(binaryStream));
    CHK_RET(ConnInfoPack(binaryStream));
    CHK_RET(TransResPack(binaryStream));
    CHK_RET(BufferInfoPack(binaryStream, locBufferInfos_));
    binaryStream.Dump(sendData_);
    u32 sendSize = sendData_.size();

    // 发送数据包尺寸
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(reinterpret_cast<u8 *>(&sendSize), sizeof(sendSize));
    EXCEPTION_HANDLE_END
    HCCL_INFO("[CcuTransport::%s] Send size[%u] of data success. [%zu] bytes sent.",
        __func__, sendSize, sizeof(sendSize));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvDataSize()
{
    // 接收数据包尺寸
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(reinterpret_cast<u8 *>(&exchangeDataSize_), sizeof(exchangeDataSize_));
    EXCEPTION_HANDLE_END
    HCCL_INFO("[CcuTransport::%s] Receive size[%u] of data success. [%zu] bytes received.",
        __func__, exchangeDataSize_, sizeof(exchangeDataSize_));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::SendConnAndTransInfo()
{
    // 当前socket失败会抛异常，需要统一整改
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(reinterpret_cast<u8 *>(sendData_.data()), sendData_.size());
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvConnAndTransInfo()
{
    recvData_.resize(exchangeDataSize_);
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(reinterpret_cast<u8 *>(recvData_.data()), recvData_.size());
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvDataProcess()
{
    Hccl::BinaryStream binaryStream(recvData_);
    CHK_RET(HandshakeMsgUnpack(binaryStream));
    CHK_RET(ConnInfoUnpackProc(binaryStream));
    CHK_RET(TransResUnpackProc(binaryStream));
    rmtBufferInfos_.clear();
    remoteUserMemTag_.clear();
    CHK_RET(BufferInfoUnpack(binaryStream));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::SendTransInfo()
{
    Hccl::BinaryStream binaryStream;
    TransResPack(binaryStream);
    binaryStream.Dump(sendTrans_);
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(reinterpret_cast<u8 *>(sendTrans_.data()), sendTrans_.size());
    EXCEPTION_HANDLE_END
    exchangeDataSize_ = sendTrans_.size();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvTransInfo()
{
    recvTrans_.resize(exchangeDataSize_);
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(reinterpret_cast<u8 *>(recvTrans_.data()), recvTrans_.size());
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvTransInfoProcess()
{
    Hccl::BinaryStream binaryStream(recvTrans_);
    TransResUnpackProc(binaryStream);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::HandshakeMsgPack(Hccl::BinaryStream &binaryStream)
{
    binaryStream << attr_.handshakeMsg;
    HCCL_INFO("[CcuTransport][%s] start pack handshakeMsg, attr.handshakeMsg.size[%zu]",
        __func__, attr_.handshakeMsg.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::ConnInfoPack(Hccl::BinaryStream &binaryStream) const
{
    std::vector<char> dtoData{};
    CHK_RET(ccuConnection_->Serialize(dtoData));
    binaryStream << dtoData;
    HCCL_INFO("[CcuTransport][%s] start pack connInfo, dtoData.size[%u]", __func__, dtoData.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::TransResPack(Hccl::BinaryStream &binaryStream)
{
    const uint32_t locCkesSize = locRes_.ckes.size();
    binaryStream << locCkesSize;
    const uint32_t locCkeSize = locRes_.ckes.size();
    for (uint32_t i = 0; i < locCkeSize; i++) {
        binaryStream << locRes_.ckes[i];
    }

    const uint32_t locXnsSize = locRes_.xns.size();
    binaryStream << locXnsSize;
    for (uint32_t i = 0; i < locRes_.xns.size(); i++) {
        binaryStream << locRes_.xns[i];
    }
    HCCL_INFO("Send ckesSize[%u], xnsSize[%u]", locCkesSize, locXnsSize);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::BufferInfoPack(Hccl::BinaryStream &binaryStream, std::vector<CclBufferInfo> &bufferVec) const
{
    u32 locBufferNum = bufferVec.size();
    binaryStream << locBufferNum;
    for (u32 pos = 0; pos < locBufferNum; ++pos) {
        bufferVec[pos].Pack(binaryStream);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::HandshakeMsgUnpack(Hccl::BinaryStream &binaryStream)
{
    binaryStream >> rmtHandshakeMsg_;

    if (attr_.handshakeMsg.size() != rmtHandshakeMsg_.size()) {
        HCCL_ERROR("handshakeMsg size=%u is not equal to rmt=%u",
            attr_.handshakeMsg.size(), rmtHandshakeMsg_.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("[CcuTransport][%s] start unpack handshakeMsg", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::ConnInfoUnpackProc(Hccl::BinaryStream &binaryStream) const
{
    std::vector<char> dtoData{};
    binaryStream >> dtoData;
    CHK_RET(ccuConnection_->Deserialize(dtoData));
    HCCL_INFO("[CcuTransport][%s] start unpack connInfo, dtoData.size[%u]", __func__, dtoData.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::TransResUnpackProc(Hccl::BinaryStream &binaryStream)
{
    uint32_t resSize{0};
    binaryStream >> resSize;
    rmtRes_.ckes.clear();
    for (uint32_t i = 0; i < resSize; i++) {
        uint32_t cke{0};
        binaryStream >> cke;
        rmtRes_.ckes.push_back(cke);
    }
    HCCL_INFO("Recv ckesSize[%u]", resSize);

    binaryStream >> resSize;
    rmtRes_.xns.clear();
    for (uint32_t i = 0; i < resSize; i++) {
        uint32_t xn{0};
        binaryStream >> xn;
        rmtRes_.xns.push_back(xn);
    }
    HCCL_INFO("Recv xnsSize[%u]", resSize);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::BufferInfoUnpack(Hccl::BinaryStream &binaryStream)
{
    u32 rmtBufferNum{0};
    binaryStream >> rmtBufferNum;
    CHK_PRT_RET(rmtBufferNum == 0 || rmtBufferNum > MAX_BUFFER_NUM,
        HCCL_ERROR("[CcuTransport][BufferInfoUnpack] rmtBufferNum[%u] is zero or exceeds limit[%u]",
            rmtBufferNum, MAX_BUFFER_NUM),
        HCCL_E_PARA);
    HCCL_INFO("[CcuTransport][BufferInfoUnpack] rmtBufferNum[%u]", rmtBufferNum);
    for (u32 pos = 0; pos < rmtBufferNum; ++pos) {
        CclBufferInfo rmtBufferInfo{};
        rmtBufferInfo.Unpack(binaryStream);
        rmtBufferInfos_.push_back(rmtBufferInfo);
        remoteUserMemTag_.push_back(rmtBufferInfo.memTag);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::SendFinish()
{
    sendFinishMsg_ = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(reinterpret_cast<u8 *>(sendFinishMsg_.data()), FINISH_MSG_SIZE);
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(reinterpret_cast<u8 *>(recvFinishMsg_.data()), FINISH_MSG_SIZE);
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::CheckFinish()
{
    const std::string sendFinishMsgStr(sendFinishMsg_.begin(), sendFinishMsg_.end());
    const std::string recvFinishMsgStr(recvFinishMsg_.begin(), recvFinishMsg_.end());
    if (sendFinishMsgStr != recvFinishMsgStr) {
        HCCL_ERROR("[CcuTransport][RecvFinish]msgRecv[%s] and msgSend[%s] are not equal",
            recvFinishMsgStr.c_str(), sendFinishMsgStr.c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::ReleaseTransRes()
{
    for (uint32_t i = 0; i < ckesRes_.size(); i++) {
        if (ckesRes_[i].empty()) {
            continue;
        }
        auto ret = CcuDevMgrImp::ReleaseCke(devLogicId_, dieId_, ckesRes_[i]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[CcuTransport][%s] release ckes failed but passed, "
                "devLogicId[%d] dieId[%u].", __func__, devLogicId_, dieId_);
        }
    }
    ckesRes_.clear();

    for (uint32_t i = 0; i < xnsRes_.size(); i++) {
        if (xnsRes_[i].empty()) {
            continue;
        }
        auto ret = CcuDevMgrImp::ReleaseXn(devLogicId_, dieId_, xnsRes_[i]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[CcuTransport][%s] release xns failed but passed, "
                "devLogicId[%d] dieId[%u].", __func__, devLogicId_, dieId_);
        }
    }
    xnsRes_.clear();

    return HcclResult::HCCL_SUCCESS;
}

uint32_t CcuTransport::GetDieId() const
{
    return dieId_;
}

uint32_t CcuTransport::GetChannelId() const
{
    return ccuConnection_->GetChannelId();
}

HcclResult CcuTransport::GetLocCkeByIndex(const uint32_t index, uint32_t &locCkeId) const
{
    CHK_PRT_RET(locRes_.ckes.empty(),
        HCCL_ERROR("[CcuTransport][%s] failed, local resources is empty.",
            __func__),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(index >= locRes_.ckes.size(),
        HCCL_ERROR("[CcuTransport][%s] failed, index[%u] is larger than size[%u].",
            __func__, index, locRes_.ckes.size()),
        HcclResult::HCCL_E_PARA);

    locCkeId = locRes_.ckes[index];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetLocXnByIndex(const uint32_t index, uint32_t &locXnId) const
{
    CHK_PRT_RET(locRes_.xns.empty(),
        HCCL_ERROR("[CcuTransport][%s] failed, local resources is empty.",
            __func__),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(index >= locRes_.xns.size(),
        HCCL_ERROR("[CcuTransport][%s] failed, index[%u] is larger than size[%u].",
            __func__, index, locRes_.xns.size()),
        HcclResult::HCCL_E_PARA);

    locXnId = locRes_.xns[index];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtCkeByIndex(const uint32_t index, uint32_t &rmtCkeId) const
{
    CHK_PRT_RET(rmtRes_.ckes.empty(),
        HCCL_ERROR("[CcuTransport][%s] failed, local resources is empty.",
            __func__),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(index >= rmtRes_.ckes.size(),
        HCCL_ERROR("[CcuTransport][%s] failed, index[%u] is larger than size[%u].",
            __func__, index, rmtRes_.ckes.size()),
        HcclResult::HCCL_E_PARA);

    rmtCkeId = rmtRes_.ckes[index];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtXnByIndex(const uint32_t index, uint32_t &rmtXnId) const
{
    CHK_PRT_RET(rmtRes_.xns.empty(),
        HCCL_ERROR("[CcuTransport][%s] failed, local resources is empty.",
            __func__),
        HcclResult::HCCL_E_PARA);
    
    CHK_PRT_RET(index >= rmtRes_.xns.size(),
        HCCL_ERROR("[CcuTransport][%s] failed, index[%u] is larger than size[%u].",
            __func__, index, rmtRes_.xns.size()),
        HcclResult::HCCL_E_PARA);

    rmtXnId = rmtRes_.xns[index];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetLocBuffer(CclBufferInfo &bufferInfo, const uint32_t &bufNum) const
{
    (void)bufNum;
    bufferInfo = locBufferInfos_[0];
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtBuffer(CclBufferInfo &bufferInfo, const uint32_t &bufNum) const
{
    (void)bufNum;
    bufferInfo = rmtBufferInfos_[0];
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::GetCkeNum(uint32_t &ckeNum) const
{
    ckeNum = locRes_.ckes.size();
    return HcclResult::HCCL_SUCCESS;
}

CcuTransport::~CcuTransport()
{
    (void)ReleaseTransRes();
}

std::string CcuTransport::Describe() const
{
    std::string description = "";

    description = Hccl::StringFormat("DieId: %u, ", dieId_);
    description += transStatus_.Describe();
    description += Hccl::StringFormat(", LocRes: {%u Ckes, %u Xns}, ", locRes_.ckes.size(),
                                locRes_.xns.size());
    description += Hccl::StringFormat("RmtRes: {%u Ckes, %u Xns}, ", rmtRes_.ckes.size(),
                                rmtRes_.xns.size());
    description += Hccl::StringFormat("CkesRes size: %u, ", ckesRes_.size());
    description += Hccl::StringFormat("XnsRes size: %u.", xnsRes_.size());
    return description;
}

HcclResult CcuTransport::Describe(std::string &dfxMsg)
{
    CHK_RET(ccuConnection_->Describe(dfxMsg));
    return HcclResult::HCCL_SUCCESS;
}

void CcuTransport::Clean()
{
    transStatus_ = TransStatus::INIT;
    sendData_.clear();
    ccuConnection_->Clean();
}

HcclResult CcuTransport::GetUserRemoteMem(CommMem **remoteMem, char ***memTags, uint32_t *memNum)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    uint32_t userMemCount = rmtBufferInfos_.size() - 1; // 默认 cclBuffer 数量为1，后续出现1的含义也是 cclBufferNum
    auto cacheBuilder = [](Hccl::RemoteMemCtx<CclBufferInfo> &remoteMemCtx, uint32_t index) {
        auto &rmtBuffer = remoteMemCtx.rmtBufferVec[index + 1];
        remoteMemCtx.remoteUserMems[index].type = rmtBuffer.type;
        remoteMemCtx.remoteUserMems[index].addr = reinterpret_cast<void *>(rmtBuffer.addr);
        remoteMemCtx.remoteUserMems[index].size = rmtBuffer.size;
    };
    Hccl::RemoteMemCtx<CclBufferInfo> remoteMemCtx{
        userMemCount, cacheValid_, rmtBufferInfos_, remoteUserMemTag_, remoteUserMems_, tagCopies_, tagPointers_,
        cacheBuilder, remoteMem, memTags, memNum};
    CHK_RET(Hccl::GetRemoteUserMem(remoteMemCtx));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::CheckSocketStatus()
{
    CHK_PTR_NULL(socket_);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while(true) {
        EXCEPTION_HANDLE_BEGIN
        Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
        if (socketStatus == Hccl::SocketStatus::OK) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            HCCL_INFO("[CcuTransport][%s] success, elapsed[%lld]ms, retryCount[%u]",
                __func__, elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout ||
            socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            HCCL_ERROR("[CcuTransport][%s] channel connect timeout after %lld sec, elapsed[%lld]ms, retryCount[%u]",
                __func__, timeout, elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        EXCEPTION_HANDLE_END
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::UpdateMemInfo(std::vector<CcuTransport::CclBufferInfo> &bufferVecTemp)
{
    if (bufferVecTemp.size() == 0) {
        HCCL_WARNING("[CcuTransport][UpdateMemInfo] bufferNum is 0.");
        return HCCL_SUCCESS;
    }
    uint32_t totalBufferNum = locBufferInfos_.size() + bufferVecTemp.size();
    if (UNLIKELY(totalBufferNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR("[CcuTransport][UpdateMemInfo] totalBufferNum[%u] exceeds limit[%u]", totalBufferNum, MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[CcuTransport][UpdateMemInfo] bufferNum[%zu]", bufferVecTemp.size());
    sendData_.clear();
    Hccl::BinaryStream sendStream;
    CHK_RET(BufferInfoPack(sendStream, bufferVecTemp));
    sendStream.Dump(sendData_);
    u32 sendSize = sendData_.size();
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(reinterpret_cast<u8 *>(&sendSize), sizeof(sendSize));
    EXCEPTION_HANDLE_END
    HCCL_INFO("[CcuTransport][UpdateMemInfo] Send size[%u] of data success. [%zu] bytes sent.",
        sendSize, sizeof(sendSize));
    CHK_RET(CheckSocketStatus());
    CHK_RET(RecvDataSize());
    CHK_RET(CheckSocketStatus());
    CHK_RET(SendConnAndTransInfo());
    CHK_RET(CheckSocketStatus());
    CHK_RET(RecvConnAndTransInfo());
    CHK_RET(CheckSocketStatus());
    Hccl::BinaryStream recvStream(recvData_);
    CHK_RET(BufferInfoUnpack(recvStream));
    locBufferInfos_.insert(locBufferInfos_.end(), bufferVecTemp.begin(), bufferVecTemp.end());
    // 流程中已有新增内存数量判断，故执行到此位置一定存在新增内存，需要将标识置位false，使得再次调用GetUserRemoteMem时重新构造缓存
    cacheValid_ = false;
    return HcclResult::HCCL_SUCCESS;
}
} // namespace hcomm