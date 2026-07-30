/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_ASSIGN_H
#define HCOMM_CCU_REPRESENTATION_ASSIGN_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepAssign : public CcuRepBase {
public:
    explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Variable &varA, uint64_t immediate);
    explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address &addrA, uint64_t immediate);
    explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address &addrA, const Variable &varA);
    explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address &addrB, const Address &addrA);
    explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Variable &varB, const Variable &varA);

    bool          Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string   Describe() override;
    Address     GetAddrA();
    Address     GetAddrB();
    Variable    GetVarA();
    Variable    GetVarB();
    uint64_t    GetImmed();
    AssignSubType  GetSubType();
private:
    void SetCommonInfo();
    CcuInsGeneratorBase* insGenPtr{nullptr};
    AssignSubType subType{AssignSubType::INVALID};
    uint64_t immediate{0};

    Variable varA;
    Variable varB;

    Address addrA;
    Address addrB;
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_ASSIGN_H