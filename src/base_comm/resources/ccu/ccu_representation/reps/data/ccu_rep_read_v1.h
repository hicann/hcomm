/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_READ_H
#define HCOMM_CCU_REPRESENTATION_READ_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepRead : public CcuRepBase {

public:
    CcuRepRead(CcuInsGeneraterBase* insGenPtr, const ChannelHandle channel, LocalAddr loc, RemoteAddr rem, Variable len, CompletedEvent sem,
               uint16_t mask);
    CcuRepRead(CcuInsGeneraterBase* insGenPtr, const ChannelHandle channel, LocalAddr loc, RemoteAddr rem, Variable len, uint16_t dataType,
               uint16_t opType, CompletedEvent sem, uint16_t mask);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
    uint16_t GetLocAddrId() { return  loc.addr.Id(); }
    uint16_t GetLocTokenId() { return  loc.token.Id(); }
    uint16_t GetRemAddrId() { return  rem.addr.Id(); }
    uint16_t GetRemTokenId() { return  rem.token.Id(); }
    uint16_t GetLenId() { return  len.Id(); }
    uint16_t GetSemId() { return  sem.Id(); }
    uint32_t GetChannelId() { return channelId; }

    ChannelHandle GetChannel() { return channel; }
    LocalAddr GetLoc() { return loc; }
    RemoteAddr GetRem() { return rem; }
    Variable GetLen() { return len; }
    CompletedEvent GetSem() { return sem; }
    uint16_t GetMask() { return mask; }
    uint16_t GetDataType() { return dataType; }
    uint16_t GetOpType() { return opType; }
    uint16_t GetReduceFlag() { return reduceFlag; }

private:
    CcuInsGeneraterBase* insGenPtr{nullptr};
    ChannelHandle channel;
    uint32_t channelId{0};

    LocalAddr   loc;
    RemoteAddr   rem;
    Variable len;

    CompletedEvent sem;
    uint16_t  mask{0};

    uint16_t dataType{0};
    uint16_t opType{0};
    uint16_t reduceFlag{0};

};

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_READ_H