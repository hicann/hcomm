/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_BUFREDUCE_H
#define HCOMM_CCU_REPRESENTATION_BUFREDUCE_H

#include <vector>

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepBufReduce : public CcuRepBase {
public:
    CcuRepBufReduce(const std::vector<CcuBuf> &mem, uint16_t count, uint16_t dataType, uint16_t outputDataType,
                    uint16_t opType, CompletedEvent sem, const CcuRep::Variable &len, uint16_t mask = 1);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
    const std::vector<CcuBuf>& GetMem() { return mem; }
    uint16_t GetCount() const { return count; }
    uint16_t GetDataType() const { return dataType; }
    uint16_t GetOutputDataType() const { return outputDataType; }
    uint16_t GetOpType() const { return opType; }
    uint16_t GetXnLengthId() { return xnIdLength_.Id(); }
    uint16_t GetMask() const { return mask; }
    uint16_t GetSemId() { return sem.Id(); }

private:
    std::vector<CcuBuf> mem;
    uint16_t               count;
    uint16_t               dataType;
    uint16_t               outputDataType;
    uint16_t               opType;
    CompletedEvent         sem;
    CcuRep::Variable       xnIdLength_;
    uint16_t               mask{0};
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_BUFREDUCE_H