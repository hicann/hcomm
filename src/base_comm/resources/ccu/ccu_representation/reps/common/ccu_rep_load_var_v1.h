/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation load var header file
 * Create: 2025-04-22
 */

#ifndef HCOMM_CCU_REP_LOAD_VAR_H
#define HCOMM_CCU_REP_LOAD_VAR_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepLoadVar : public CcuRepBase {
public:
    CcuRepLoadVar(CcuInsGeneraterBase* insGenPtr, const Variable &src, const Variable &var, uint32_t num = 1);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;

    CcuRep::Variable GetVar() { return var; }
    CcuRep::Variable GetSrc() { return src; }
    uint32_t GetNum() { return num; }
    uint16_t GetMask() { return mask; }

private:
    CcuInsGeneraterBase* insGeneratorPtr_{nullptr};
    Variable src;
    Variable var;
    uint32_t num;
    uint16_t mask{1};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REP_LOAD_VAR_H