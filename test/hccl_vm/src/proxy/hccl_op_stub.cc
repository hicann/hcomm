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
#include "hccl/hccl_types.h"
#include "sim_common_defs.h"
#include "hccl_proxy_common.h"
#include "hccl_rank_graph.h"
#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "db_sim_op_db_ops.h"
#include "db_sim_runner_ops.h"


extern "C" uint8_t GetOpExpansionMode();

namespace {
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
        const void* inputBuf, uint32_t inputSize,
        const void* outputBuf, uint32_t outputSize,
        const std::vector<uint8_t>& details,
        uint32_t root,
        uint32_t rankSize,
        uint32_t srcRank,
        uint32_t dstRank,
        const std::vector<uint8_t>& extInfo = {})
    {
        (void) cmdType;
        sim::OpDetailTab opDetailTab;
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

        sim::OpMemInfoTab opMemInfoTab;
        opMemInfoTab.id = 0;
        opMemInfoTab.inputAddr = reinterpret_cast<uint64_t>(inputBuf);
        opMemInfoTab.inputSize = inputSize;
        opMemInfoTab.outputAddr = reinterpret_cast<uint64_t>(outputBuf);
        opMemInfoTab.outputSize = outputSize;
        opMemInfoTab.cclAddr = 0;
        opMemInfoTab.cclSize = 0;

        int ret = sim::InsertOpDetailAndMem(opDetailTab, opMemInfoTab);
        if (ret != 0) {
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
    uint32_t inputSize = static_cast<uint32_t>(inDataSize * sendCount * rankSize);
    uint32_t outputSize = static_cast<uint32_t>(outDataSize * recvCount * rankSize);

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
    uint32_t inCountTotal = 0;
    uint32_t outCountTotal = 0;
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        inCountTotal += ((uint64_t*)sendCounts)[rank];
        outCountTotal += ((uint64_t*)recvCounts)[rank];
    }
    uint32_t inputSize = static_cast<uint32_t>(inDataSize * inCountTotal);
    uint32_t outputSize = static_cast<uint32_t>(outDataSize * outCountTotal);

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
    uint32_t inputSize = static_cast<uint32_t>(dataSize * sendCount);
    uint32_t outputSize = static_cast<uint32_t>(dataSize * sendCount * rankSize);

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
    uint32_t size = static_cast<uint32_t>(dataSize * count);

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
    uint32_t size = static_cast<uint32_t>(dataSize * count);
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
    uint32_t inputValueSize = 0;
    if (curRank == root) {
        inputValueSize = static_cast<uint32_t>(dataSize * recvCount * rankSize);
    }
    uint32_t outputValueSize = static_cast<uint32_t>(dataSize * recvCount);
 
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
    uint32_t size = static_cast<uint32_t>(dataSize * count);

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
    uint32_t inputSize = static_cast<uint32_t>(dataSize * recvCount * rankSize);
    uint32_t outputSize = static_cast<uint32_t>(dataSize * recvCount);

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
 
HcclResult _Z18GetRunSideIsDeviceRb(bool &isDeviceSide)
{
    isDeviceSide = false;
    return HcclResult::HCCL_SUCCESS;
}
 
#ifdef __cplusplus
}
#endif  // __cplusplus
