/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include "exception_util.h"
#include "ccu_ins_generater_base.h"
#include "internal_exception.h"

namespace hcomm {
namespace CcuRep {

CcuRepRemPostSem::CcuRepRemPostSem(CcuInsGeneraterBase* insGenPtr, const ChannelHandle channel, uint16_t semIndex, uint16_t mask)
    : insGenPtr(insGenPtr), channel(channel), semIndex(semIndex), mask(mask)
{
    type       = CcuRepType::REM_POST_SEM;
    instrCount = insGenPtr->GetInstrCount(type);
}

bool CcuRepRemPostSem::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    insGenPtr->CcuRepRemPostSemTranslate(ccuKernel, instr, this, dep);
    CHK_PRT_THROW((instrId > UINT16_MAX - instrCount),
                        HCCL_ERROR("[CcuRepRemPostSem::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", instrId, instrCount),
                          Hccl::InternalException, "integer overflow");
    instrId += instrCount;

    return translated;
}

std::string CcuRepRemPostSem::Describe()
{
    return Hccl::StringFormat("Post, Use semIndex[%u] and mask[%04x]", semIndex, mask);
}

}; // namespace CcuRep
}; // namespace hcomm