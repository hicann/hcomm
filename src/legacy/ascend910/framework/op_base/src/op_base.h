/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_BASE_H
#define OP_BASE_H

#include <vector>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include <hccl/hccl_types.h>

#include "op_base_pub.h"
#include "hccl_comm_pub.h"
// ltm指定config路径
#include "common/src/config.h"
#include "../common/src/topo/topoinfo_detect.h"
#include "op_base_v2.h"

using HcclOpInfoCtx = struct HcclInfoTag {
    HcclCommPtr pComm;
    hccl::HcclCommParams params;
    hccl::RankTable_t rankTable;
    bool cloudFlag = false; // cloudFlag为0即实验室场景,cloudFlag为1则为云场景
    bool isUsed;
    std::mutex opGroupMapMutex;
    std::unordered_map<std::string, std::shared_ptr<hccl::hcclComm>> opGroup2CommMap;
    std::map<std::string, std::shared_ptr<hccl::TopoInfoDetect>> hcclCommTopoInfoDetectServer;
    std::map<std::string, std::shared_ptr<hccl::TopoInfoDetect>> hcclCommTopoInfoDetectAgent;
    HcclInfoTag() : isUsed(false) {}

    ~HcclInfoTag()
    {
        pComm = nullptr;
        opGroup2CommMap.clear();
        hcclCommTopoInfoDetectServer.clear();
        hcclCommTopoInfoDetectAgent.clear();
    }
};

constexpr uint32_t MAX_HCOM_NUM = 3U;

HcclOpInfoCtx& GetHcclExistDeviceOpInfoCtx(void);

HcclOpInfoCtx& GetHcclOpInfoCtx(void);

HcclResult InitOtherInfo(hccl::HcclCommParams& params, const char* rankTable);

HcclResult CallMsprofReportHostApi(
    hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count, HcclDataType dataType,
    const std::string& tag);

HcclResult ReduceScatterLoop(
    const std::string& tag, void* inputPtr, void* outputPtr, const u64& count, HcclDataType dataType, HcclReduceOp op,
    hccl::hcclComm* hcclComm, rtStream_t stream);

HcclResult HcclGetOpBasedMemSize(const HcclCMDType& opType, u64& size, const HcomCollOpInfo& opInfo);

HcclResult ReduceLoop(
    const std::string& tag, void* inputPtr, void* outputPtr, const u64 count, HcclDataType dataType, HcclReduceOp op,
    const u32 root, hccl::hcclComm* hcclComm, rtStream_t stream);

HcclResult HcclGatherAlltoAllV(HcomGatherAllToAllVParams params, HcclComm comm, aclrtStream stream);

HcclResult RunGather(u64* sendCounts, u64* sdispls, void* sendDevBuf, GatherPara& gatherPara);

void GatherMemCopyThread(void* baseAddr, u64 offset, std::vector<u64>& addrInfo, OpBaseMemPara memCpyPara);

HcclResult HcclGetCommAll(uint32_t ndev, int32_t* devices, HcclComm* comms);

HcclResult
GetDeviceComm(uint32_t ndev, const HcclRootInfo& rootHandle, const s32 rank, const s32 logicDeviceId, HcclComm& comm);

HcclResult SetOverFlowAddr(hccl::hcclComm* hcclComm);

HcclResult HcclGetCommHandle(const char* commName, std::shared_ptr<hccl::hcclComm>& comm);

HcclResult CheckScatterInputPara(HcclComm comm, void* recvBuf);

HcclResult HcclMc2ComResourceByTiling(HcclComm comm, uint32_t* pVersion, void* mc2Tiling, rtStream_t& aicpuStream);

HcclResult HcclCreateComResourceByComm(
    HcclComm comm, u32 streamMode, bool isOpbaseMode, void** commContext, bool isMC2 = false,
    void* mc2Tiling = nullptr);

HcclResult HcclDeviceRefresh(s32& deviceLogicId);

HcclResult HcclBatchSendRecvGroup(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);

HcclResult HcclSetIfProfile(void);

void HcclResetIfProfile(void);

void PrintCountsAndDispls(const u32 length, const void* counts, const void* displs, const std::string& tag);

void CheckCountsAndDispls(const u32 length, const void* counts, const void* displs, const std::string& tag);

HcclResult GetCaptureInfo(aclrtStream stream, aclmdlRICaptureStatus& captureStatus, uint64_t& modelId, bool& isCapture);

HcclResult HcclGetInitTilingList(const void* mc2Tiling, const void* p[], uint32_t& cnt);

HcclResult HcclCommInitCollComm(uint32_t rank, void** commV2, const HcclCommConfig* config, HcclComm* comm);
HcclResult HcclMc2ComOpResCtx(
    HcclComm comm, uint8_t opType, HcclDataType srcDataType, HcclDataType dstDataType, HcclReduceOp reduceType,
    uint64_t count, char* algConfig, uint32_t commEngine, rtStream_t& aicpuStream);

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult
HcclCommInitClusterInfoMemConfig(const char* rankTableString, uint32_t rank, HcclCommConfig* config, HcclComm* comm);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
HcclResult HcclCommStateNotify(HcclComm comm, HcclCommStatePhase state);

static inline HcclResult HcclCommResumePostCallback(HcclComm comm)
{
    return HcclCommStateNotify(comm, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_RESUME_POST);
}
#endif
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // OP_BASE_H
