/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_BUFREAD_H
#define HCOMM_CCU_REPRESENTATION_BUFREAD_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepBufRead : public CcuRepBase {
public:
    CcuRepBufRead(const ChannelHandle channel, RemoteAddr src, CcuBuf dst, Variable len, CompletedEvent sem,
                  uint32_t mask);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;

    uint16_t GetSrcAddrId() { return src.addr.Id(); }
    uint16_t GetSrcTokenId() { return src.token.Id(); }
    uint16_t GetDstAddrId() { return dst.Id(); }
    uint16_t GetLenId() { return len.Id(); }
    uint16_t GetSemId() { return sem.Id(); }
    uint32_t GetMask() const { return mask; }
    uint32_t GetChannelId() const { return channelId; }
private:
    ChannelHandle channel;
    uint32_t channelId{0};

    RemoteAddr    src;
    CcuBuf dst;
    Variable  len;

    CompletedEvent sem;
    uint32_t   mask{0};
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_BUFREAD_H