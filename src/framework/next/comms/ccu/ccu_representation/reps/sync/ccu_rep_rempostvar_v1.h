/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_REMPOSTVAR_H
#define HCOMM_CCU_REPRESENTATION_REMPOSTVAR_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepRemPostVar : public CcuRepBase {
public:
    CcuRepRemPostVar(Variable param, const ChannelHandle channel, uint16_t paramIndex, uint16_t semIndex,
                       uint16_t mask);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
    uint32_t    GetMask() const { return mask; }
    uint32_t    GetRmtXnId() const { return rmtXnId; }
    uint32_t    GetRmtCkeId() const { return rmtCkeId; }
    uint16_t    GetParamIndex() const { return paramIndex; }
    Variable    GetParam() { return param; }
    uint32_t    GetChannelId() const { return channelId; }

private:
    Variable                      param;
    ChannelHandle                 channel;
    uint16_t                      paramIndex{0};
    uint16_t                      semIndex{0};
    uint16_t                      mask{0};
    uint32_t                      rmtXnId{0};
    uint32_t                      rmtCkeId{0};
    uint32_t                      channelId{0};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_REMPOSTVAR_H