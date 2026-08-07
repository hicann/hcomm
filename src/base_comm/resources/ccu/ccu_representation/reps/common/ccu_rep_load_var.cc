/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation load var file
 * Create: 2025-04-22
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

    CcuRepLoadVar::CcuRepLoadVar(CcuInsGeneratorBase* insGenPtr, const Variable& src, const Variable& var, uint32_t num)
        : insGeneratorPtr_(insGenPtr),
          src(src),
          var(var),
          num(num)
    {
        type = CcuRepType::LOAD_VAR;
        instrCount = insGeneratorPtr_->GetInstrCount(type);
    }

    bool CcuRepLoadVar::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        Hccl::CHECK_NULLPTR(instr, "[CcuRepLoadVar::Translate] instr is nullptr!");
        this->instrId = instrId;
        translated = true;
        CHK_RET_THROW(
            Hccl::CcuApiException,
            Hccl::StringFormat(
                "[CcuRepLoadVar][%s] failed to translate repLoadVar for instrId[%u] ", __func__, instrId),
            insGeneratorPtr_->CcuRepLoadVarTranslate(ccuKernel, instr, instrId, this, dep));
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepLoadVar::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            Hccl::InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoadVar::Describe()
    {
        return Hccl::StringFormat("Load Var([%llu], [%u], [%u])", src.Id(), var.Id(), num);
    }

}; // namespace CcuRep
}; // namespace hcomm
