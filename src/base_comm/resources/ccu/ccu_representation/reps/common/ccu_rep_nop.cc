/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generater_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

using namespace Hccl;

CcuRepNop::CcuRepNop(CcuInsGeneraterBase* insGeneratorPtr) : insGeneratorPtr_(insGeneratorPtr)
{
    type       = CcuRepType::NOP;
    instrCount = 1;
}

bool CcuRepNop::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    CHK_RET_THROW(Hccl::CcuApiException,
        Hccl::StringFormat("[CcuRepNop][%s] failed to translate repNop for instrId[%u] ", __func__, instrId),
            insGeneratorPtr_->CcuRepNopTranslate(ccuKernel, instr, instrId, this, dep));

    instrId += instrCount;

    return translated;
}

std::string CcuRepNop::Describe()
{
    return Hccl::StringFormat("Nop");
}

}; // namespace CcuRep
}; // namespace hcomm