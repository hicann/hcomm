/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_NOP_H
#define HCOMM_CCU_REPRESENTATION_NOP_H

#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepNop : public CcuRepBase {
public:
    CcuRepNop(CcuInsGeneraterBase* insGeneratorPtr);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;

private:
    CcuInsGeneraterBase* insGeneratorPtr_{nullptr};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_NOP_H