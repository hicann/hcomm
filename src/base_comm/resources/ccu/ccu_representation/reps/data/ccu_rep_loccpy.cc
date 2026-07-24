/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include "string_util.h"
#include "ccu_ins_generater_v1.h"
#include "ccu_ins_generater_base.h"
#include "ccu_kernel.h"

#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

CcuRepLocCpy::CcuRepLocCpy(CcuInsGeneraterBase* insGenPtr, LocalAddr dst, LocalAddr src, Variable len, CompletedEvent sem, uint16_t mask)
    : insGenPtr(insGenPtr), dst(dst), src(src), len(len), sem(sem), mask(mask)
{
    type       = CcuRepType::LOCAL_CPY;
    instrCount = insGenPtr->GetInstrCount(type);
    useCcuBuffer = false;
}

CcuRepLocCpy::CcuRepLocCpy(CcuInsGeneraterBase* insGenPtr, LocalAddr dst, LocalAddr src, Variable len, uint16_t dataType, uint16_t opType, CompletedEvent sem,
                           uint16_t mask)
    : insGenPtr(insGenPtr), dst(dst), src(src), len(len), sem(sem), mask(mask), dataType(dataType), opType(opType) 
{
    type       = CcuRepType::LOCAL_REDUCE;
    instrCount = insGenPtr->GetInstrCount(type);
    reduceFlag = 1;
    // A5和A6都走环回
    useCcuBuffer = false;
}

CcuRepLocCpy::CcuRepLocCpy(CcuInsGeneraterBase* insGenPtr, LocalAddr dst, LocalAddr src, Variable len,
                           const std::vector<CcuBuf> &bufs, CompletedEvent sem, uint16_t mask)
    : insGenPtr(insGenPtr), dst(dst), src(src), len(len), bufs(bufs), sem(sem), mask(mask)
{
    type       = CcuRepType::LOCAL_CPY;
    instrCount = insGenPtr->GetInstrCount(type);
    useCcuBuffer = true;
}

void CcuRepLocCpy::ValidateInsGeneratorForLocCpy()
{
    CcuInsGeneraterV1* tmpPtrV1 = dynamic_cast<CcuInsGeneraterV1*>(insGenPtr);
    if (tmpPtrV1 && useCcuBuffer) {
        // 使用了A6场景的ms中转搬运
        Hccl::THROW<Hccl::CcuApiException>("Cannot translate CcuRepLocCpy for A5 when useCcuBuffer is true!");
    }
}

uint16_t CcuRepLocCpy::GetFirstBufId()
{
    if (bufs.size() == 0) {
        Hccl::THROW<Hccl::CcuApiException>("The length of CcuBuffer is 0!");
    }
    return bufs[0].Id();
}
    
uint16_t CcuRepLocCpy::GetUsedBufNum()
{
    return bufs.size();
}

bool CcuRepLocCpy::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    ValidateInsGeneratorForLocCpy();

    this->instrId = instrId;
    translated    = true;

    insGenPtr->CcuRepLocCpyTranslate(ccuKernel, instr, this, dep);
    instrId += instrCount;

    return translated;
}

std::string CcuRepLocCpy::Describe()
{
    return Hccl::StringFormat(
        "Read LocalAddr[%u] to LocalAddr[%u], length[%u], set sem[%u] with mask[%04x], dataType[%u], opType[%u]",
        src.addr.Id(), dst.addr.Id(), len.Id(), sem.Id(), mask, dataType, opType);
}

}; // namespace CcuRep
}; // namespace hcomm