/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_BUFWRITE_H
#define HCOMM_CCU_REPRESENTATION_BUFWRITE_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepBufWrite : public CcuRepBase {
public:
    CcuRepBufWrite(const ChannelHandle channel, CcuBuf src, RemoteAddr dst, Variable len, CompletedEvent sem,
                   uint16_t mask);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;

    uint16_t GetSrcId() { return src.Id(); }
    uint16_t GetDstAddrId() { return dst.addr.Id(); }
    uint16_t GetDstTokenId() { return dst.token.Id(); }
    uint16_t GetLenId() { return len.Id(); }
    uint16_t GetSemId() { return sem.Id(); }
    uint32_t GetMask() const { return mask; }
    uint32_t GetChannelId() const { return channelId; }
private:
    ChannelHandle channel;
    uint32_t channelId{0};

    CcuBuf src;
    RemoteAddr    dst;
    Variable  len;

    CompletedEvent sem;
    uint16_t   mask{0};
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_BUFWRITE_H