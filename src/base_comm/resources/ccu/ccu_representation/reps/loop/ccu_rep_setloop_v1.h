/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_SETLOOP_H
#define HCOMM_CCU_REPRESENTATION_SETLOOP_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepSetLoop : public CcuRepBase {
    public:
        CcuRepSetLoop(
            CcuInsGeneratorBase* insGeneratorPtr, const Variable& loopParam, const Executor& executor,
            const Variable& var);

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

    private:
        CcuInsGeneratorBase* insGeneratorPtr_{nullptr};

    public:
        Variable loopParam;
        Executor executor;
        Variable var;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_SETLOOP_H
