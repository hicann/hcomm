/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation store implementation file
 * Create: 2025-03-21
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

CcuRepStore::CcuRepStore(CcuInsGeneraterBase* insGenPtr, const Variable &var, uint64_t addr, uint32_t num) :
    insGeneratorPtr_(insGenPtr), var(var), addr(addr), num(num)
{
    type       = CcuRepType::STORE;
    instrCount = insGeneratorPtr_->GetInstrCount(type);
}

bool CcuRepStore::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId    = instrId;
    translated       = true;
    CHK_RET_THROW(Hccl::CcuApiException,
        Hccl::StringFormat("[CcuRepStore][%s] failed to translate repStore for instrId[%u] ", __func__, instrId),
            insGeneratorPtr_->CcuRepStoreTranslate(ccuKernel, instr, instrId, this, dep));
    instrId += instrCount;

    return translated;
}

std::string CcuRepStore::Describe()
{
    return Hccl::StringFormat("Store([%u], [%llu], [%u])", var.Id(), addr, num);
}

}; // namespace CcuRep
}; // namespace hcomm