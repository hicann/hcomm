/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Description: ccu executor -- trans locms to locms
 * Author: caiyifan
 */

#include "trans_loc_ms_to_loc_ms_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V1(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMSTOLOCMS_CODE, TransLocMSToLocMSExecutor);

void TransLocMSToLocMSExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        dstMSId_     = instr_.v1.transLocMSToLocMS.dstMSId & 0x7FFF;
        dstDieId_    = instr_.v1.transLocMSToLocMS.dstMSId >> 15;
        srcMSId_     = instr_.v1.transLocMSToLocMS.srcMSId & 0x7FFF;
        srcDieId_    = instr_.v1.transLocMSToLocMS.srcMSId >> 15;
        lengthXnId_  = instr_.v1.transLocMSToLocMS.lengthXnId;
        channelId_   = instr_.v1.transLocMSToLocMS.channelId;
        clearType_   = instr_.v1.transLocMSToLocMS.clearType;
        lengthEn_    = instr_.v1.transLocMSToLocMS.lengthEn;
        setCKEId_    = instr_.v1.transLocMSToLocMS.setCKEId;
        setCKEMask_  = instr_.v1.transLocMSToLocMS.setCKEMask;
        waitCKEId_   = instr_.v1.transLocMSToLocMS.waitCKEId;
        waitCKEMask_ = instr_.v1.transLocMSToLocMS.waitCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void TransLocMSToLocMSExecutor::Process(CcuResourceManager &ccuResMgr)
{
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        srcMSId_ += msOffset;
        dstMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG("locCcu[{}:{}], Get ms offset = [{:04x}], cke offset = [{:04x}]",
               rankId_, dieId_, msOffset, ckeOffset);
    }
    HCCL_VM_DEBUG("locCcu[{}:{}] Trans data "
           "from locSrcMsId[{}] to locDstMsId[{}], "
           "with lengthXnId[{}] transLength[{}].",
           rankId_, dieId_, srcMSId_, dstMSId_, lengthXnId_, transLength_);
    ccuResMgr.TransMSToMS(rankId_, dieId_, rankId_, dieId_, srcMSId_, dstMSId_, transLength_);
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMSToLocMSExecutor::RunV1()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransLocMsToLocMs");
}

void TransLocMSToLocMSExecutor::Run()
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

std::string TransLocMSToLocMSExecutor::Describe()
{
    return HcclSim::StringFormat("ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMS[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], "
                              "Set CKE[%u:%04x], "
                              "clearType[%u], lengthEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        srcMSId_ / 0x8000,
        srcMSId_ % 0x8000,
        dstMSId_ / 0x8000,
        dstMSId_ % 0x8000,
        lengthXnId_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_);
}
