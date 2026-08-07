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

#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepLoadArg::CcuRepLoadArg(
        CcuInsGeneratorBase* insGenPtr, const Variable& var, uint16_t argId, uint16_t fullArgId)
        : insGeneratorPtr_(insGenPtr),
          var(var),
          argId(argId),
          fullArgId(fullArgId)
    {
        type = CcuRepType::LOAD_ARG;
        instrCount = insGeneratorPtr_->GetInstrCount(type);
    }

    bool CcuRepLoadArg::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_RET_THROW(
            Hccl::CcuApiException,
            Hccl::StringFormat(
                "[CcuRepLoadArg][%s] failed to translate repLoadArg for instrId[%u] ", __func__, instrId),
            insGeneratorPtr_->CcuRepLoadArgTranslate(ccuKernel, instr, instrId, this, dep));

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoadArg::Describe()
    {
        return Hccl::StringFormat("Variable[%u] = Arg[%u] (slot=%u)", var.Id(), fullArgId, argId);
    }

}; // namespace CcuRep
}; // namespace hcomm
