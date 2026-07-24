/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "OP_STUB"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "acl/acl_rt.h"
#include "dtype_common.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "sim_common_defs.h"
#include "hccl_proxy_common.h"
#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "db_sim_op_db_ops.h"
#include "db_sim_runner_ops.h"


extern "C" uint8_t GetOpExpansionMode();

namespace {
    static std::vector<uint8_t> BuildBatchSendRecvA2AExtInfo(uint32_t itemNum)
    {
        std::vector<uint8_t> extInfo(sizeof(itemNum));
        std::memcpy(extInfo.data(), &itemNum, sizeof(itemNum));
        return extInfo;
    }

    static std::vector<uint8_t> BuildVOpExtInfo(uint32_t rankSize, uint64_t localCount,
        const uint64_t *counts, const uint64_t *displs)
    {
        const size_t arrayByteSize = static_cast<size_t>(rankSize) * sizeof(uint64_t);
        std::vector<uint8_t> extInfo(sizeof(localCount) + arrayByteSize * 2);
        size_t offset = 0;
        std::memcpy(extInfo.data() + offset, &localCount, sizeof(localCount));
        offset += sizeof(localCount);
        std::memcpy(extInfo.data() + offset, counts, arrayByteSize);
        offset += arrayByteSize;
        std::memcpy(extInfo.data() + offset, displs, arrayByteSize);
        return extInfo;
    }

    static std::vector<uint8_t> BuildMatrixExtInfo(const uint64_t *matrix, uint32_t matrixElementCount)
    {
        const size_t matrixByteSize = static_cast<size_t>(matrixElementCount) * sizeof(uint64_t);
        std::vector<uint8_t> extInfo(sizeof(matrixElementCount) + matrixByteSize);
        std::memcpy(extInfo.data(), &matrixElementCount, sizeof(matrixElementCount));
        std::memcpy(extInfo.data() + sizeof(matrixElementCount), matrix, matrixByteSize);
        return extInfo;
    }

    static std::vector<uint8_t> BuildOpDetailsV1(uint64_t count,
                                             uint16_t dataType, uint32_t reduceOpType, uint32_t reduceOp,
                                             HcclCMDType opType) {
        OpDetails details;
        std::memset(&details, 0, sizeof(details));
        details.opType = static_cast<uint16_t>(opType);
        details.dataType = dataType;
        details.reduceType = reduceOp;
        
        details.opV1.count = count;
        details.opV1.dataType = dataType;
        details.opV1.reduceOpType = reduceOpType;

        std::vector<uint8_t> blob(sizeof(OpDetails));
        std::memcpy(blob.data(), &details, sizeof(OpDetails));
        return blob;
    }

    static std::vector<uint8_t> BuildOpDetailsV2(uint64_t sendCount, uint16_t sendDataType,
                                             uint64_t recvCount, uint16_t recvDataType, uint16_t extInfo,
                                             HcclCMDType opType, uint16_t dataType, uint32_t reduceOp) {
        OpDetails details;
        std::memset(&details, 0, sizeof(details));
        details.opType = static_cast<uint16_t>(opType);
        details.dataType = dataType;
        details.reduceType = reduceOp;
        
        details.opV2.sendCount = sendCount;
        details.opV2.sendDataType = sendDataType;
        details.opV2.recvCount = recvCount;
        details.opV2.recvDataType = recvDataType;
        details.opV2.extInfo = extInfo;

        std::vector<uint8_t> blob(sizeof(OpDetails));
        std::memcpy(blob.data(), &details, sizeof(OpDetails));
        return blob;
    }

    static uint32_t GetDeviceType()
    {
       DevType devType;
       auto ret = hrtGetDeviceType(devType);
       if(ret != HcclResult::HCCL_SUCCESS){
         return ret;
       }
       return static_cast<uint32_t>(devType);
    }

    int RecordOpDbInfo(HcclCMDType cmdType,
        uint32_t rankId, uint64_t streamId,
        const void* inputBuf, uint64_t inputSize,
        const void* outputBuf, uint64_t outputSize,
        const std::vector<uint8_t>& details,
        uint32_t root,
        uint32_t rankSize,
        uint32_t srcRank,
        uint32_t dstRank,
        const std::vector<uint8_t>& extInfo = {})
    {
        (void) cmdType;
        sim::OpDetailTab opDetailTab{};
        opDetailTab.id = 0;
        opDetailTab.pid = getpid();
        opDetailTab.rankId = rankId;
        opDetailTab.opIter = 0;
        opDetailTab.syncIter = 0;
        opDetailTab.streamId = streamId;
        opDetailTab.root = root;
        opDetailTab.opExpansionMode = GetOpExpansionMode();
        opDetailTab.devType = GetDeviceType();
        opDetailTab.rankSize = rankSize;
        opDetailTab.srcRank = srcRank;
        opDetailTab.dstRank = dstRank;
        opDetailTab.opDetail = details;
        opDetailTab.opExtInfo = extInfo;

        sim::OpMemInfoTab opMemInfoTab{};
        opMemInfoTab.id = 0;
        opMemInfoTab.inputAddr = reinterpret_cast<uint64_t>(inputBuf);
        opMemInfoTab.inputSize = inputSize;
        opMemInfoTab.outputAddr = reinterpret_cast<uint64_t>(outputBuf);
        opMemInfoTab.outputSize = outputSize;
        opMemInfoTab.cclAddr = 0;
        opMemInfoTab.cclSize = 0;

        if (sim::InsertOpDetailAndMem(opDetailTab, opMemInfoTab) != 0) {
            HCCL_VM_ERROR("insert op detail+mem failed");
            return -1;
        }
        return 0;
    }
}

using namespace HcclSim;
 
#ifdef __cplusplus
extern "C" {
#endif

uint8_t GetOpExpansionMode()
{
    sim::SimOpExpansionMode mode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_RESERVED;
    const char *expanEnv = std::getenv("HCCL_OP_EXPANSION_MODE");
    if (expanEnv == nullptr) {
        HCCL_VM_INFO("HCCL_OP_EXPANSION_MODE env is not set, use default value: [{}]", static_cast<uint8_t>(mode));
        return static_cast<uint8_t>(mode);
    }

    std::string modeStr(expanEnv);
    if (modeStr == "AI_CPU") {
        mode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU;
    } else if (modeStr == "AIV") {
        mode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AIV;
    } else if (modeStr == "CCU_SCHED" || modeStr == "CCU_MS") {
        mode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    }

    HCCL_VM_INFO("HCCL_OP_EXPANSION_MODE = [{}]", expanEnv);
    return static_cast<uint8_t>(mode);
}

HcclResult HcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
    HcclComm comm, aclrtStream stream)
{
    const uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());
    const uint32_t rankSize = sim::GetRankSize();
    if (comm == nullptr || stream == nullptr || rankSize == 0 || curRank >= rankSize ||
        destRank >= rankSize || destRank == curRank || (count != 0 && sendBuf == nullptr)) {
        HCCL_VM_ERROR("invalid HcclSend parameters: rank={}, rankSize={}, destRank={}", curRank, rankSize, destRank);
        return HcclResult::HCCL_E_PARA;
    }
    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataSize = count * typeSize;
    const auto details = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_SEND);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_SEND, curRank, reinterpret_cast<uint64_t>(stream),
            sendBuf, dataSize, nullptr, 0, details, 0, rankSize, curRank, destRank) != 0) {
        return HcclResult::HCCL_E_PARA;
    }
    using Func = HcclResult (*)(void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    const auto func = reinterpret_cast<Func>(dlsym(RTLD_NEXT, __func__));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclSend failed: {}", dlerror());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(sendBuf, count, dataType, destRank, comm, stream);
}

HcclResult HcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
    HcclComm comm, aclrtStream stream)
{
    const uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());
    const uint32_t rankSize = sim::GetRankSize();
    if (comm == nullptr || stream == nullptr || rankSize == 0 || curRank >= rankSize ||
        srcRank >= rankSize || srcRank == curRank || (count != 0 && recvBuf == nullptr)) {
        HCCL_VM_ERROR("invalid HcclRecv parameters: rank={}, rankSize={}, srcRank={}", curRank, rankSize, srcRank);
        return HcclResult::HCCL_E_PARA;
    }
    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataSize = count * typeSize;
    const auto details = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_RECEIVE);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_RECEIVE, curRank, reinterpret_cast<uint64_t>(stream),
            nullptr, 0, recvBuf, dataSize, details, 0, rankSize, srcRank, curRank) != 0) {
        return HcclResult::HCCL_E_PARA;
    }
    using Func = HcclResult (*)(void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    const auto func = reinterpret_cast<Func>(dlsym(RTLD_NEXT, __func__));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclRecv failed: {}", dlerror());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(recvBuf, count, dataType, srcRank, comm, stream);
}

HcclResult HcclBatchSendRecv(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum,
    HcclComm comm, aclrtStream stream)
{
    const uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());
    const uint32_t rankSize = sim::GetRankSize();
    if (sendRecvInfo == nullptr || comm == nullptr || stream == nullptr || rankSize == 0 ||
        curRank >= rankSize || static_cast<uint64_t>(itemNum) != static_cast<uint64_t>(rankSize) * 2) {
        HCCL_VM_ERROR("invalid HcclBatchSendRecv parameters: rank={}, rankSize={}, itemNum={}",
            curRank, rankSize, itemNum);
        return HcclResult::HCCL_E_PARA;
    }

    const uint64_t peerCount = sendRecvInfo[0].count;
    const HcclDataType dataType = sendRecvInfo[0].dataType;
    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t peerBytes = peerCount * typeSize;
    const uint64_t totalBytes = peerBytes * rankSize;

    std::vector<const HcclSendRecvItem *> sends(rankSize, nullptr);
    std::vector<const HcclSendRecvItem *> recvs(rankSize, nullptr);
    for (uint32_t i = 0; i < itemNum; ++i) {
        const auto &item = sendRecvInfo[i];
        if (item.remoteRank >= rankSize || item.count != peerCount || item.dataType != dataType ||
            (peerCount != 0 && item.buf == nullptr)) {
            HCCL_VM_ERROR("HcclBatchSendRecv only supports uniform all-to-all items");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        auto *slots = item.sendRecvType == HCCL_SEND ? &sends :
            (item.sendRecvType == HCCL_RECV ? &recvs : nullptr);
        if (slots == nullptr || (*slots)[item.remoteRank] != nullptr) {
            HCCL_VM_ERROR("HcclBatchSendRecv has an invalid or duplicate item for peer {}", item.remoteRank);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        (*slots)[item.remoteRank] = &item;
    }
    if (sends[0] == nullptr || recvs[0] == nullptr) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uintptr_t sendBase = reinterpret_cast<uintptr_t>(sends[0]->buf);
    const uintptr_t recvBase = reinterpret_cast<uintptr_t>(recvs[0]->buf);
    for (uint32_t peer = 0; peer < rankSize; ++peer) {
        const uint64_t offset = peer * peerBytes;
        if (sends[peer] == nullptr || recvs[peer] == nullptr ||
            reinterpret_cast<uintptr_t>(sends[peer]->buf) != sendBase + offset ||
            reinterpret_cast<uintptr_t>(recvs[peer]->buf) != recvBase + offset) {
            HCCL_VM_ERROR("HcclBatchSendRecv only supports contiguous all-to-all buffers, peer={}", peer);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
    }

    const auto details = BuildOpDetailsV1(peerCount, dataType, 0, 0,
        HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    const auto extInfo = BuildBatchSendRecvA2AExtInfo(itemNum);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, curRank, reinterpret_cast<uint64_t>(stream),
            reinterpret_cast<void *>(sendBase), totalBytes, reinterpret_cast<void *>(recvBase), totalBytes,
            details, 0, rankSize, curRank, curRank, extInfo) != 0) {
        return HcclResult::HCCL_E_PARA;
    }
    using Func = HcclResult (*)(HcclSendRecvItem *, uint32_t, HcclComm, aclrtStream);
    const auto func = reinterpret_cast<Func>(dlsym(RTLD_NEXT, __func__));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclBatchSendRecv failed: {}", dlerror());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(sendRecvInfo, itemNum, comm, stream);
}

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAlltoAll called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("sendType = {}", static_cast<int>(sendType));
    HCCL_VM_INFO("recvType = {}", static_cast<int>(recvType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);
 
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    uint32_t inDataSize = 0;
    if (sim::GetDataTypeSize(sendType, inDataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t outDataSize = 0;
    if (sim::GetDataTypeSize(recvType, outDataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(inDataSize) * sendCount * rankSize;
    uint64_t outputSize = static_cast<uint64_t>(outDataSize) * recvCount * rankSize;

    // Use BuildOpDetailsV2 for AlltoAll
    auto alltoallDetails = BuildOpDetailsV2(
        sendCount, sendType, recvCount, recvType, 0, HcclCMDType::HCCL_CMD_ALLTOALL, sendType, 0);
    std::vector<uint8_t> alltoallExtInfo;
    uint32_t alltoallCount = rankSize * rankSize;
    alltoallExtInfo.resize(sizeof(uint32_t) + alltoallCount * sizeof(uint64_t));
    std::memcpy(alltoallExtInfo.data(), &alltoallCount, sizeof(uint32_t));
    for (uint32_t i = 0; i < alltoallCount; i++) {
        std::memcpy(alltoallExtInfo.data() + sizeof(uint32_t) + i * sizeof(uint64_t), &sendCount, sizeof(uint64_t));
    }
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_ALLTOALL, curRank, reinterpret_cast<uint64_t>(stream),
                   sendBuf, inputSize, recvBuf, outputSize, alltoallDetails,
                   0, rankSize, curRank, curRank, alltoallExtInfo) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAlltoAllFunc = HcclResult (*)(
        const void *, uint64_t, HcclDataType, const void *, uint64_t, HcclDataType, HcclComm, aclrtStream);
    HcclAlltoAllFunc hcclAlltoAllFunc = reinterpret_cast<HcclAlltoAllFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAlltoAllFunc != nullptr) {
        return hcclAlltoAllFunc(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAlltoAllVC called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCountMatrix = {:p}", sendCountMatrix);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendType = {}", static_cast<int>(sendType));
    HCCL_VM_INFO("recvType = {}", static_cast<int>(recvType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    const uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());
    const uint32_t rankSize = sim::GetRankSize();
    if (rankSize == 0 || curRank >= rankSize || sendCountMatrix == nullptr) {
        HCCL_VM_ERROR("invalid AlltoAllVC rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    const auto *matrix = static_cast<const uint64_t *>(sendCountMatrix);
    uint64_t inputCount = 0;
    uint64_t outputCount = 0;
    const size_t rowOffset = static_cast<size_t>(curRank) * static_cast<size_t>(rankSize);
    for (uint32_t peerRank = 0; peerRank < rankSize; ++peerRank) {
        inputCount += matrix[rowOffset + peerRank];
        outputCount += matrix[static_cast<size_t>(peerRank) * rankSize + curRank];
    }

    uint32_t sendDataSize = 0;
    if (sim::GetDataTypeSize(sendType, sendDataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t recvDataSize = 0;
    if (sim::GetDataTypeSize(recvType, recvDataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    const uint64_t inputSize = inputCount * sendDataSize;
    const uint64_t outputSize = outputCount * recvDataSize;

    const auto alltoallvcDetails = BuildOpDetailsV2(
        inputCount, sendType, outputCount, recvType, 0, HcclCMDType::HCCL_CMD_ALLTOALLVC, sendType, 0);
    const uint32_t matrixElementCount = rankSize * rankSize;
    const auto alltoallvcExtInfo = BuildMatrixExtInfo(matrix, matrixElementCount);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_ALLTOALLVC, curRank, reinterpret_cast<uint64_t>(stream),
            sendBuf, inputSize, recvBuf, outputSize, alltoallvcDetails,
            0, rankSize, curRank, curRank, alltoallvcExtInfo) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAlltoAllVCFunc = HcclResult (*)(
        const void *, const void *, HcclDataType, const void *, HcclDataType, HcclComm, aclrtStream);
    const auto hcclAlltoAllVCFunc = reinterpret_cast<HcclAlltoAllVCFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAlltoAllVCFunc != nullptr) {
        return hcclAlltoAllVCFunc(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
    }

    HCCL_VM_ERROR("dlsym failed");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}
 
HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAlltoAllV called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendType = {}", static_cast<int>(sendType));
    HCCL_VM_INFO("recvType = {}", static_cast<int>(recvType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);
 
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    // auto inDataSize = DATA_TYPE_SIZE_MAP.at(sendType);
    // auto outDataSize = DATA_TYPE_SIZE_MAP.at(recvType);
    uint32_t inDataSize = 0;
    if (sim::GetDataTypeSize(sendType, inDataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t outDataSize = 0;
    if (sim::GetDataTypeSize(recvType, outDataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inCountTotal = 0;
    uint64_t outCountTotal = 0;
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        inCountTotal += ((uint64_t*)sendCounts)[rank];
        outCountTotal += ((uint64_t*)recvCounts)[rank];
    }
    uint64_t inputSize = static_cast<uint64_t>(inDataSize) * inCountTotal;
    uint64_t outputSize = static_cast<uint64_t>(outDataSize) * outCountTotal;

    auto alltoallvDetails = BuildOpDetailsV2(
        static_cast<uint64_t>(inCountTotal), sendType, static_cast<uint64_t>(outCountTotal), recvType, 0,
        HcclCMDType::HCCL_CMD_ALLTOALLV, sendType, 0);

    std::vector<uint64_t> sendCountMatrix(rankSize * rankSize, 0);
    for (uint32_t j = 0; j < rankSize; j++) {
        sendCountMatrix[curRank * rankSize + j] = ((uint64_t*)sendCounts)[j];
    }

    for (uint32_t j = 0; j < rankSize; j++) {
        sendCountMatrix[j * rankSize + curRank] = ((uint64_t*)recvCounts)[j];
    }

    std::vector<uint8_t> alltoallvExtInfo;
    uint32_t alltoallvCount = rankSize * rankSize;
    alltoallvExtInfo.resize(sizeof(uint32_t) + alltoallvCount * sizeof(uint64_t));
    std::memcpy(alltoallvExtInfo.data(), &alltoallvCount, sizeof(uint32_t));
    std::memcpy(alltoallvExtInfo.data() + sizeof(uint32_t), sendCountMatrix.data(), alltoallvCount * sizeof(uint64_t));
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_ALLTOALLV, curRank, reinterpret_cast<uint64_t>(stream),
                   sendBuf, inputSize, recvBuf, outputSize, alltoallvDetails,
                   0, rankSize, curRank, curRank, alltoallvExtInfo) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);
 
    using HcclAlltoAllVFunc = HcclResult (*)(const void *,
        const void *,
        const void *,
        HcclDataType,
        const void *,
        const void *,
        const void *,
        HcclDataType,
        HcclComm,
        aclrtStream);
    HcclAlltoAllVFunc hcclAlltoAllVFunc = reinterpret_cast<HcclAlltoAllVFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAlltoAllVFunc != nullptr) {
        return hcclAlltoAllVFunc(
            sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}
 
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAllGather called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);
 
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    // auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(dataSize) * sendCount;
    uint64_t outputSize = static_cast<uint64_t>(dataSize) * sendCount * rankSize;

    // Use BuildOpDetailsV1 for AllGather
    auto allGatherDetails = BuildOpDetailsV1(
        sendCount, dataType, 0, 0, HcclCMDType::HCCL_CMD_ALLGATHER);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_ALLGATHER, curRank, reinterpret_cast<uint64_t>(stream),
                   sendBuf, inputSize, recvBuf, outputSize, allGatherDetails,
                   0, rankSize, curRank, curRank) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);
 
    using HcclAllGatherFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, HcclComm, aclrtStream);
    HcclAllGatherFunc hcclAllGatherFunc = reinterpret_cast<HcclAllGatherFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAllGatherFunc != nullptr) {
        return hcclAllGatherFunc(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf,
    const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAllGatherV called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCounts = {:p}", recvCounts);
    HCCL_VM_INFO("recvDispls = {:p}", recvDispls);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    const uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());
    const uint32_t rankSize = sim::GetRankSize();
    if (rankSize == 0 || curRank >= rankSize) {
        HCCL_VM_ERROR("invalid AllGatherV rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    if (recvCounts == nullptr || recvDispls == nullptr) {
        HCCL_VM_ERROR("invalid AllGatherV count or displacement array");
        return HcclResult::HCCL_E_PARA;
    }

    const auto *recvCountValues = static_cast<const uint64_t *>(recvCounts);
    const auto *recvDisplValues = static_cast<const uint64_t *>(recvDispls);
    uint64_t totalRecvCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        totalRecvCount += recvCountValues[rank];
    }

    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t inputSize = sendCount * dataSize;
    const uint64_t outputSize = totalRecvCount * dataSize;

    const auto allGatherVExtInfo = BuildVOpExtInfo(rankSize, sendCount, recvCountValues, recvDisplValues);
    const auto allGatherVDetails = BuildOpDetailsV1(
        sendCount, dataType, 0, 0, HcclCMDType::HCCL_CMD_ALLGATHER_V);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_ALLGATHER_V, curRank, reinterpret_cast<uint64_t>(stream),
            sendBuf, inputSize, recvBuf, outputSize, allGatherVDetails,
            0, rankSize, curRank, curRank, allGatherVExtInfo) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAllGatherVFunc = HcclResult (*)(
        void *, uint64_t, void *, const void *, const void *, HcclDataType, HcclComm, aclrtStream);
    const auto hcclAllGatherVFunc = reinterpret_cast<HcclAllGatherVFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAllGatherVFunc != nullptr) {
        return hcclAllGatherVFunc(
            sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
    }

    HCCL_VM_ERROR("dlsym failed");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}
 
HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclBroadcast called with parameters:");
    HCCL_VM_INFO("buf = {:p}", buf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("root = {}", root);
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);
 
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    // auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * count;

    // Use BuildOpDetailsV1 for Broadcast
    auto broadcastDetails = BuildOpDetailsV1(
        count, dataType, 0, 0, HcclCMDType::HCCL_CMD_BROADCAST);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_BROADCAST, curRank, reinterpret_cast<uint64_t>(stream),
                   buf, size, buf, size, broadcastDetails,
                   root, rankSize, curRank, curRank) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);
 
    using HcclBroadcastFunc = HcclResult (*)(void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    HcclBroadcastFunc hcclBroadcastFunc = reinterpret_cast<HcclBroadcastFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclBroadcastFunc != nullptr) {
        return hcclBroadcastFunc(buf, count, dataType, root, comm, stream);
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}
 
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAllReduce called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("op = {}", static_cast<int>(op));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);
 
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    // auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * count;
    // Use BuildOpDetailsV1 for AllReduce
    auto allReduceDetails = BuildOpDetailsV1(
        count, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op), HcclCMDType::HCCL_CMD_ALLREDUCE);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_ALLREDUCE, curRank, reinterpret_cast<uint64_t>(stream),
                   sendBuf, size, recvBuf, size, allReduceDetails,
                   0, rankSize, curRank, curRank) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAddreduceFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    HcclAddreduceFunc hcclAddreduceFunc = reinterpret_cast<HcclAddreduceFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAddreduceFunc != nullptr) {
        return hcclAddreduceFunc(sendBuf, recvBuf, count, dataType, op, comm, stream);
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}
 
HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclScatter called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("root = {}", root);
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);
 
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    // auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputValueSize = 0;
    if (curRank == root) {
        inputValueSize = static_cast<uint64_t>(dataSize) * recvCount * rankSize;
    }
    uint64_t outputValueSize = static_cast<uint64_t>(dataSize) * recvCount;
 
    const void* inputPtr = (curRank == root) ? sendBuf : nullptr;
    // Use BuildOpDetailsV1 for Scatter
    auto scatterDetails = BuildOpDetailsV1(
        recvCount, dataType, 0, 0, HcclCMDType::HCCL_CMD_SCATTER);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_SCATTER, curRank, reinterpret_cast<uint64_t>(stream),
                   inputPtr, inputValueSize, recvBuf, outputValueSize, scatterDetails,
                   root, rankSize, curRank, curRank) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);
 
    using HcclScatterFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    HcclScatterFunc hcclScatterFunc = reinterpret_cast<HcclScatterFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclScatterFunc != nullptr) {
        return hcclScatterFunc(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}
 
HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size)
{
    HCCL_VM_INFO("HcclGetHcclBufferNew called with parameters: buffer= {:p}, {}", *buffer, *size);
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
 
    using HcclGetHcclBufferFunc = HcclResult (*)(HcclComm, void**, uint64_t*);
    auto hcclGetHcclBufferFunc = reinterpret_cast<HcclGetHcclBufferFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclGetHcclBufferFunc != nullptr) {
        auto ret = hcclGetHcclBufferFunc(comm, buffer, size);
        sim::UpdateOpMemCclBuffer(reinterpret_cast<uint64_t>(*buffer), *size);
        HCCL_VM_INFO("get rank{} ccl buffer= {:p}, {}", curRank, *buffer, *size);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}
 
HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclReduce called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("reduce op = {}", static_cast<int>(op));
    HCCL_VM_INFO("root = {}", root);
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);
 
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    // auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * count;

    // Use BuildOpDetailsV1 for Reduce
    auto reduceDetails = BuildOpDetailsV1(
        count, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op), HcclCMDType::HCCL_CMD_REDUCE);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_REDUCE, curRank, reinterpret_cast<uint64_t>(stream),
                   sendBuf, size, recvBuf, size, reduceDetails,
                   root, rankSize, curRank, curRank) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);
 
    using HcclReduceFunc =
        HcclResult (*)(void *, void *, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream);
    auto hcclReduceFunc = reinterpret_cast<HcclReduceFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclReduceFunc != nullptr) {
        return hcclReduceFunc(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}
 
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclReduceScatter called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("reduce op = {}", static_cast<int>(op));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);
 
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(dataSize) * recvCount * rankSize;
    uint64_t outputSize = static_cast<uint64_t>(dataSize) * recvCount;

    // Use BuildOpDetailsV1 for ReduceScatter
    auto reduceScatterDetails = BuildOpDetailsV1(
        recvCount, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op), HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, curRank, reinterpret_cast<uint64_t>(stream),
                   sendBuf, inputSize, recvBuf, outputSize, reduceScatterDetails,
                   0, rankSize, curRank, curRank) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclReduceScatterFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    auto hcclReduceScatterFunc = reinterpret_cast<HcclReduceScatterFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclReduceScatterFunc != nullptr) {
        return hcclReduceScatterFunc(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls,
    void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclReduceScatterV called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCounts = {:p}", sendCounts);
    HCCL_VM_INFO("sendDispls = {:p}", sendDispls);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("reduce op = {}", static_cast<int>(op));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    const uint32_t curRank = static_cast<uint32_t>(sim::GetCurrRankId());
    const uint32_t rankSize = sim::GetRankSize();
    if (rankSize == 0 || curRank >= rankSize) {
        HCCL_VM_ERROR("invalid ReduceScatterV rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    if (sendCounts == nullptr || sendDispls == nullptr) {
        HCCL_VM_ERROR("invalid ReduceScatterV count or displacement array");
        return HcclResult::HCCL_E_PARA;
    }

    const auto *sendCountValues = static_cast<const uint64_t *>(sendCounts);
    const auto *sendDisplValues = static_cast<const uint64_t *>(sendDispls);
    uint64_t totalSendCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        totalSendCount += sendCountValues[rank];
    }

    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t inputSize = totalSendCount * dataSize;
    const uint64_t outputSize = recvCount * dataSize;

    const auto reduceScatterVExtInfo = BuildVOpExtInfo(rankSize, recvCount, sendCountValues, sendDisplValues);
    const auto reduceScatterVDetails = BuildOpDetailsV1(
        recvCount, dataType, 0, op, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    if (RecordOpDbInfo(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, curRank, reinterpret_cast<uint64_t>(stream),
            sendBuf, inputSize, recvBuf, outputSize, reduceScatterVDetails,
            0, rankSize, curRank, curRank, reduceScatterVExtInfo) != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclReduceScatterVFunc = HcclResult (*)(void *, const void *, const void *, void *, uint64_t,
        HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    const auto hcclReduceScatterVFunc = reinterpret_cast<HcclReduceScatterVFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclReduceScatterVFunc != nullptr) {
        return hcclReduceScatterVFunc(
            sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }

    HCCL_VM_ERROR("dlsym failed");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}
 
HcclResult _Z18GetRunSideIsDeviceRb(bool &isDeviceSide)
{
    isDeviceSide = false;
    return HcclResult::HCCL_SUCCESS;
}
 
#ifdef __cplusplus
}
#endif  // __cplusplus
