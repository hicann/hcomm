/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include "string_util.h"
#include "ccu_ins_generater_v1.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generater_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

CcuRepBufReduce::CcuRepBufReduce(CcuInsGeneraterBase* insGenPtr, const std::vector<CcuBuf> &mem, uint16_t count, uint16_t dataType,
                                 uint16_t outputDataType, uint16_t opType, CompletedEvent sem, const CcuRep::Variable &len,
                                 uint16_t mask)
    : insGenPtr(insGenPtr), mem(mem), count(count), dataType(dataType), outputDataType(outputDataType), opType(opType), sem(sem),
      xnIdLength_(len), mask(mask)
{
    type       = CcuRepType::BUF_REDUCE;
    instrCount = 1;
}

bool CcuRepBufReduce::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    instrCount = insGenPtr->GetInstrCount(type);
    insGenPtr->CcuRepBufReduceTranslate(ccuKernel, instr, this);

    instrId += instrCount;

    return translated;
}

std::string CcuRepBufReduce::Describe()
{
    return Hccl::StringFormat("Reduce");
}

}; // namespace CcuRep
}; // namespace hcomm