/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOOP_H
#define HCOMM_CCU_REPRESENTATION_LOOP_H

#include <memory>

#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_loopblock_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepLoop : public CcuRepBase {
public:
    explicit CcuRepLoop(CcuInsGeneratorBase* insGeneratorPtr, const std::string &label, const Variable &loopParam);
    explicit CcuRepLoop(CcuInsGeneratorBase* insGeneratorPtr, const std::string &label, 
        const Variable &loopParam, const Variable &loopIterNum, const Variable &loopGsaOffset);
    bool               Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string        Describe() override;
    const std::string &GetLabel() const;

    void                        Reference(std::shared_ptr<CcuRepLoopBlock> refRep);
    std::shared_ptr<CcuRepBase> SetLoopParam(Executor executor, Variable var);
    CcuRepLoopBlock* GetLoopBlock() { return loopBlock.get(); }
    
    Variable* GetLoopParam() { return &loopParam; }
    Variable GetLoopIterNum() {return loopIterNum;}
    Variable GetLoopGsaOffset() {return loopGsaOffset;}

private:
    void ValidateInsGeneratorForLoop();

    CcuInsGeneratorBase*             insGeneratorPtr_;
    std::string                      label;
    std::shared_ptr<CcuRepLoopBlock> loopBlock{nullptr};

    Variable loopParam;
    Variable loopIterNum;
    Variable loopGsaOffset;
    CcuInstr *instr{nullptr};

    bool supportCcuV1{false};
    bool supportCcuV2{false};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_LOOP_H