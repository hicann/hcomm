/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "string_util.h"
#include "exception_util.h"
#include "hcomm_c_adpt.h"
#include "ccu_rep_v1.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_kernel.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"

namespace hcomm {
namespace CcuRep {

CcuRepRemWaitSem::CcuRepRemWaitSem(CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, uint16_t semIndex, uint16_t mask, bool isProfiling)
    : insGenPtr(insGenPtr), channel(channel), semIndex(semIndex), mask(mask), isProfiling(isProfiling)
{
    type       = CcuRepType::REM_WAIT_SEM;
    instrCount = insGenPtr->GetInstrCount(type);
}

bool CcuRepRemWaitSem::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    insGenPtr->CcuRepRemWaitSemTranslate(ccuKernel, instr, this);

    CHK_PRT_THROW(instrId > UINT16_MAX - instrCount,
                        HCCL_ERROR("[CcuRepRemWaitSem::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", instrId, instrCount),
                          Hccl::InternalException, "integer overflow");

    instrId += instrCount;

    return translated;
}

std::string CcuRepRemWaitSem::Describe()
{
    return Hccl::StringFormat("Wait, Use semIndex[%u] and mask[%04x]", semIndex, mask);
}

}; // namespace CcuRep
}; // namespace hcomm