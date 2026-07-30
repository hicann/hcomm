/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include "ccu_ins_generator_v1.h"
#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

void CcuRepAssign::SetCommonInfo()
{
    type       = CcuRepType::ASSIGN;
    instrCount = insGenPtr->GetInstrCount(type);
}

CcuRepAssign::CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Variable &varA, uint64_t immediate)
    : insGenPtr(insGenPtr), subType(AssignSubType::IMD_TO_VARIABLE), immediate(immediate), varA(varA)
{
    SetCommonInfo();
}

CcuRepAssign::CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address &addrA, uint64_t immediate)
    : insGenPtr(insGenPtr), subType(AssignSubType::IMD_TO_ADDR), immediate(immediate), addrA(addrA)
{
    SetCommonInfo();
}

CcuRepAssign::CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address &addrA, const Variable &varA)
    : insGenPtr(insGenPtr), subType(AssignSubType::VAR_TO_ADDR), immediate(0), varA(varA), addrA(addrA)
{
    SetCommonInfo();
}

CcuRepAssign::CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address &addrB, const Address &addrA)
    : insGenPtr(insGenPtr), subType(AssignSubType::ADDR_TO_ADDR), immediate(0), addrA(addrA), addrB(addrB)
{
    SetCommonInfo();
}

CcuRepAssign::CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Variable &varB, const Variable &varA)
    : insGenPtr(insGenPtr), subType(AssignSubType::VAR_TO_VAR), immediate(0), varA(varA), varB(varB)
{
    SetCommonInfo();
}

bool CcuRepAssign::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    Hccl::CHECK_NULLPTR(instr, "[CcuRepAssign::Translate] instr is nullptr!");
    this->instrId = instrId;
    translated    = true;

    insGenPtr->CcuRepAssignTranslate(ccuKernel, instr, this, dep);
    
    CHK_PRT_THROW((instrId > UINT16_MAX - instrCount),
                        HCCL_ERROR("[CcuRepAssign::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", instrId, instrCount),
                          Hccl::InternalException, "integer overflow");
    instrId += instrCount;

    return translated;
}

std::string CcuRepAssign::Describe()
{
    switch (subType) {
        case AssignSubType::IMD_TO_VARIABLE: {
            return Hccl::StringFormat("Variable[%u] = Value[%lu]", varA.Id(), immediate);
        }
        case AssignSubType::IMD_TO_ADDR: {
            return Hccl::StringFormat("Address[%u] = Value[%lu]", addrA.Id(), immediate);
        }
        case AssignSubType::VAR_TO_ADDR: {
            return Hccl::StringFormat("Address[%u] = Variable[%u]", addrA.Id(), varA.Id());
        }
        case AssignSubType::ADDR_TO_ADDR: {
            return Hccl::StringFormat("Address[%u] = Address[%u]", addrB.Id(), addrA.Id());
        }
        case AssignSubType::VAR_TO_VAR: {
            return Hccl::StringFormat("Var[%u] = Var[%u]", varB.Id(), varA.Id());
        }
        default: {
            return Hccl::StringFormat("Invalid Assign");
        }
    }
}

Address CcuRepAssign::GetAddrA()
{
    return addrA;
}

Address CcuRepAssign::GetAddrB()
{
    return addrB;
}

Variable CcuRepAssign::GetVarA()
{
    return varA;
}

Variable CcuRepAssign::GetVarB()
{
    return varB;
}

uint64_t CcuRepAssign::GetImmed()
{
    return immediate;
}

AssignSubType CcuRepAssign::GetSubType()
{
    return subType;
}
}; // namespace CcuRep
}; // namespace hcomm