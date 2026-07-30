/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */
#include "ccu_rep_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "hcomm_c_adpt.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_kernel.h"
#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

CcuRepWrite::CcuRepWrite(CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, RemoteAddr rem, LocalAddr loc, Variable len, CompletedEvent sem,
                         uint16_t mask)
    : insGenPtr(insGenPtr), channel(channel), rem(rem), loc(loc), len(len), sem(sem), mask(mask)
{
    type = CcuRepType::WRITE;
    instrCount = insGenPtr->GetInstrCount(type);
}

CcuRepWrite::CcuRepWrite(CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, RemoteAddr rem, LocalAddr loc, Variable len,
                         uint16_t dataType, uint16_t opType, CompletedEvent sem, uint16_t mask)
    : insGenPtr(insGenPtr), channel(channel), rem(rem), loc(loc), len(len), sem(sem), mask(mask), 
      dataType(dataType), opType(opType), reduceFlag(1)
{
    type = CcuRepType::WRITE;
    instrCount = insGenPtr->GetInstrCount(type);
}

bool CcuRepWrite::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    insGenPtr->CcuRepWriteTranslate(ccuKernel, instr, this);
    instrId += instrCount;

    return translated;
}

std::string CcuRepWrite::Describe()
{
    return Hccl::StringFormat(
        "Write RemoteAddr[%u] to LocalAddr[%u], length[%u], set sem[%u] with mask[%04x], dataType[%u], opType[%u]",
        loc.addr.Id(), rem.addr.Id(), len.Id(), sem.Id(), mask, dataType, opType);
}

}; // namespace CcuRep
}; // namespace hcomm