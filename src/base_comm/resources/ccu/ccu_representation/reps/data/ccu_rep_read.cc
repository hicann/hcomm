/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_read_v1.h"
#include "ccu_rep_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "hcomm_c_adpt.h"

#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_ins_generater_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

CcuRepRead::CcuRepRead(CcuInsGeneraterBase* insGenPtr, const ChannelHandle channel, LocalAddr loc, RemoteAddr rem, Variable len, CompletedEvent sem,
                       uint16_t mask)
    : insGenPtr(insGenPtr), channel(channel), loc(loc), rem(rem), len(len), sem(sem), mask(mask)
{
    type       = CcuRepType::READ;
    instrCount = 1;
}

CcuRepRead::CcuRepRead(CcuInsGeneraterBase* insGenPtr, const ChannelHandle channel, LocalAddr loc, RemoteAddr rem, Variable len, uint16_t dataType,
                       uint16_t opType, CompletedEvent sem, uint16_t mask)
    : insGenPtr(insGenPtr), channel(channel), loc(loc), rem(rem), len(len), sem(sem), mask(mask), dataType(dataType), opType(opType),
      reduceFlag(1)
{
    type       = CcuRepType::READ;
    instrCount = insGenPtr->GetInstrCount(type);
}

bool CcuRepRead::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    insGenPtr->CcuRepReadTranslate(ccuKernel, instr, this);
    instrId += instrCount;

    return translated;
}

std::string CcuRepRead::Describe()
{
    return Hccl::StringFormat(
        "Read LocalAddr[%u] To RemoteAddr[%u], length[%u], set sem[%u] with mask[%04x], dataType[%u], opType[%u]",
        rem.addr.Id(), loc.addr.Id(), len.Id(), sem.Id(), mask, dataType, opType);
}

}; // namespace CcuRep
}; // namespace hcomm