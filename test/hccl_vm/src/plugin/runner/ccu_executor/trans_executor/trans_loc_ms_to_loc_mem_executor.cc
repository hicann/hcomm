/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Description: ccu executor -- trans locms to locmem
 * Author: caiyifan
 */

#include "trans_loc_ms_to_loc_mem_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V1(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMSTOLOCMEM_CODE, TransLocMSToLocMemExecutor);

void TransLocMSToLocMemExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        locGSAId_    = instr_.v1.transLocMSToLocMem.locGSAId;
        locXnId_     = instr_.v1.transLocMSToLocMem.locXnId;
        locMSId_     = instr_.v1.transLocMSToLocMem.locMSId & 0x7FFF;
        locDieId_    = instr_.v1.transLocMSToLocMem.locMSId >> 15;
        lengthXnId_  = instr_.v1.transLocMSToLocMem.lengthXnId;
        channelId_   = instr_.v1.transLocMSToLocMem.channelId;
        clearType_   = instr_.v1.transLocMSToLocMem.clearType;
        lengthEn_    = instr_.v1.transLocMSToLocMem.lengthEn;
        setCKEId_    = instr_.v1.transLocMSToLocMem.setCKEId;
        setCKEMask_  = instr_.v1.transLocMSToLocMem.setCKEMask;
        waitCKEId_   = instr_.v1.transLocMSToLocMem.waitCKEId;
        waitCKEMask_ = instr_.v1.transLocMSToLocMem.waitCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void TransLocMSToLocMemExecutor::Process(CcuResourceManager &ccuResMgr)
{
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto addrOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        locAddr  += addrOffset;
        locMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG("locCcu[{}:{}], Get gsa addr offset = [{:04x}], ms offset = [{}], "
                   "cke offset = [{:04x}]", rankId_, dieId_, addrOffset, msOffset, ckeOffset);
    }
    HCCL_VM_DEBUG("locCcu[{}:{}] Trans data "
           "from locMsId[{}] to locGSAId[{}] locAddr[{:x}], "
           "with lengthXnId[{}] transLength[{}], lengthEn_[{}].",
           rankId_, dieId_, locMSId_, locGSAId_, locAddr, lengthXnId_,
           transLength_, lengthEn_);
    bool ret = ccuResMgr.TransMSToMem(rankId_, dieId_, locMSId_, reinterpret_cast<void *>(locAddr), transLength_);
    if (!ret) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMSToLocMemExecutor::RunV1()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMSToLocMem");
}

void TransLocMSToLocMemExecutor::Run()
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

std::string TransLocMSToLocMemExecutor::Describe()
{
    return HcclSim::StringFormat("ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
                        "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
                        waitCKEId_, waitCKEMask_, locMSId_ / 0x8000, locMSId_ % 0x8000, locGSAId_, locXnId_, lengthXnId_,
                        channelId_, setCKEId_, setCKEMask_, clearType_, lengthEn_);
}
