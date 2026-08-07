/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation load implementation file
 * Author: zhanhaifeng
 * Create: 2025-03-21
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

    CcuRepLoad::CcuRepLoad(CcuInsGeneratorBase* insGenPtr, uint64_t addr, const Variable& var, uint32_t num)
        : insGeneratorPtr_(insGenPtr),
          var(var),
          addr(addr),
          num(num)
    {
        type = CcuRepType::LOAD;
        instrCount = insGeneratorPtr_->GetInstrCount(type); // 7: Load包含7条指令
    }

    bool CcuRepLoad::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        Hccl::CHECK_NULLPTR(instr, "[HCcuRepLoad::Translate] instr is nullptr!");
        this->instrId = instrId;
        translated = true;
        CHK_RET_THROW(
            Hccl::CcuApiException,
            Hccl::StringFormat("[CcuRepLoad][%s] failed to translate repLoad for instrId[%u] ", __func__, instrId),
            insGeneratorPtr_->CcuRepLoadTranslate(ccuKernel, instr, instrId, this, dep));
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepLoad::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", instrId,
                instrCount),
            Hccl::InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoad::Describe() { return Hccl::StringFormat("Load([%llu], [%u], [%u])", addr, var.Id(), num); }

}; // namespace CcuRep
}; // namespace hcomm
