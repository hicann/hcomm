/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Description: ccu executor -- trans loc mem to loc mem
 * Author: caiyifan
 */

#include "trans_loc_mem_to_loc_mem_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V1(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMEMTOLOCMEM_CODE, TransLocMemToLocMemExecutor);

void TransLocMemToLocMemExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        dstGSAId_    = instr_.v1.transLocMemToLocMem.dstGSAId;
        dstXnId_     = instr_.v1.transLocMemToLocMem.dstXnId;
        srcGSAId_    = instr_.v1.transLocMemToLocMem.srcGSAId;
        srcXnId_     = instr_.v1.transLocMemToLocMem.srcXnId;
        lengthXnId_  = instr_.v1.transLocMemToLocMem.lengthXnId;
        channelId_   = instr_.v1.transLocMemToLocMem.channelId;
        clearType_   = instr_.v1.transLocMemToLocMem.clearType;
        lengthEn_    = instr_.v1.transLocMemToLocMem.lengthEn;
        setCKEId_    = instr_.v1.transLocMemToLocMem.setCKEId;
        setCKEMask_  = instr_.v1.transLocMemToLocMem.setCKEMask;
        waitCKEId_   = instr_.v1.transLocMemToLocMem.waitCKEId;
        waitCKEMask_ = instr_.v1.transLocMemToLocMem.waitCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void TransLocMemToLocMemExecutor::Process(CcuResourceManager &ccuResMgr)
{
    uint64_t srcLocAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, srcGSAId_);
    uint64_t dstLocAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, dstGSAId_);
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        uint64_t gsaOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        uint16_t ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        srcLocAddr += gsaOffset;
        dstLocAddr += gsaOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG("locCcu[{}:{}], Get gsa addr offset = [{:x}], cke id offset = [{}]", rankId_, dieId_, gsaOffset, ckeOffset);
    }
    HCCL_VM_DEBUG("locCcu[{}:{}] Trans data from srcLocGSAId_[{}] srcLocAddr[{:x}] to dstLocGSAId_[{}] "
               "dstLocAddr[{:x}], with lengthXnId[{}] transLength[{}].",
        rankId_, dieId_, srcGSAId_, srcLocAddr, dstGSAId_, dstLocAddr, lengthXnId_, transLength_);
    ccuResMgr.TransMemToMem(reinterpret_cast<void *>(srcLocAddr), reinterpret_cast<void *>(dstLocAddr), transLength_, false, 0, 0);
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMemToLocMemExecutor::RunV1()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMemToLocMem");
}

void TransLocMemToLocMemExecutor::Run()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        RunV1();
        return;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

std::string TransLocMemToLocMemExecutor::Describe()
{
    return HcclSim::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMem[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], Set "
                              "CKE[%u:%04x], clearType[%u], lengthEn[%u]\n",
        waitCKEId_,
        waitCKEMask_,
        srcGSAId_,
        srcXnId_,
        dstGSAId_,
        dstXnId_,
        lengthXnId_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_);
}
