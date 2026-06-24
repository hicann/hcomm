/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_REMWAITSEM_H
#define HCOMM_CCU_REPRESENTATION_REMWAITSEM_H

#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepRemWaitSem : public CcuRepBase {
public:
    CcuRepRemWaitSem(const ChannelHandle channel, uint16_t semIndex, uint16_t mask, bool isProfiling=true);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
    uint32_t    GetMask() const { return mask; }
    uint32_t    GetId() override { return signalId; }
    uint16_t    GetChannelId() const { return channelId; }

private:
    ChannelHandle channel;
    uint16_t            semIndex{0};
    uint16_t            mask{0};
    bool                isProfiling{true};
    uint32_t            signalId{0};
    uint16_t            channelId{0};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_REMWAITSEM_H