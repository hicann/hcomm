/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include "ccu_ins_generator_v1.h"
#include "string_util.h"
#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

CcuRepBufLocRead::CcuRepBufLocRead(CcuInsGeneratorBase* insGenPtr, LocalAddr src, CcuBuf dst, Variable len, CompletedEvent sem, uint16_t mask)
    : insGenPtr(insGenPtr), src(src), dst(dst), len(len), sem(sem), mask(mask)
{
    type       = CcuRepType::BUF_LOC_READ;
    instrCount = insGenPtr->GetInstrCount(type);
}

bool CcuRepBufLocRead::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    insGenPtr->CcuRepBufLocReadTranslate(ccuKernel, instr, this, dep);
    instrId += instrCount;

    return translated;
}

std::string CcuRepBufLocRead::Describe()
{
    return Hccl::StringFormat("Read Loc Mem[%u] To CcuBuf[%u], len[%u], sem[%u], mask[%04x]",
    src.addr.Id(), dst.Id(), len.Id(), sem.Id(), mask);
}

}; // namespace CcuRep
}; // namespace hcomm