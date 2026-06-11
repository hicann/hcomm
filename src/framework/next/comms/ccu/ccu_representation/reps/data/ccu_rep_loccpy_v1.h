/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOCCPY_H
#define HCOMM_CCU_REPRESENTATION_LOCCPY_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepLocCpy : public CcuRepBase {
public:
    CcuRepLocCpy(LocalAddr dst, LocalAddr src, Variable len, CompletedEvent sem, uint16_t mask);
    CcuRepLocCpy(LocalAddr dst, LocalAddr src, Variable len, uint16_t dataType, uint16_t opType, CompletedEvent sem,
                 uint16_t mask);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
    uint16_t GetSrcAddrId() { return src.addr.Id(); }
    uint16_t GetSrcTokenId() { return src.token.Id(); }
    uint16_t GetDstAddrId() { return dst.addr.Id(); }
    uint16_t GetDstTokenId() { return dst.token.Id(); }
    uint16_t GetLenId() { return len.Id(); }
    uint16_t GetSemId() { return sem.Id(); }
    uint32_t GetMask() const { return mask; }
    uint16_t GetDataType() const { return dataType; }
    uint16_t GetOpType() const { return opType; }

private:
    LocalAddr   dst;
    LocalAddr   src;
    Variable len;

    CompletedEvent sem;
    uint16_t   mask{0};

    uint16_t dataType{0};
    uint16_t opType{0};
    uint16_t reduceFlag{0};
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOCCPY_H