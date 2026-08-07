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
        CcuRepRemPostVar(
            CcuInsGeneratorBase* insGenPtr, Variable param, const ChannelHandle channel, uint16_t paramIndex,
            uint16_t semIndex, uint16_t mask);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint32_t GetRmtXnId() { return rmtXnId; }
        uint32_t GetRmtCkeId() { return rmtCkeId; }
        uint16_t GetParamIndex() { return paramIndex; }
        Variable GetParam() { return param; }
        uint32_t GetChannelId() { return channelId; }

        ChannelHandle GetChannel() { return channel; }
        uint16_t GetSemIndex() { return semIndex; }
        uint16_t GetMask() { return mask; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        Variable param;
        ChannelHandle channel;
        uint16_t paramIndex{0};
        uint16_t semIndex{0};
        uint16_t mask{0};
        uint32_t rmtXnId{0};
        uint32_t rmtCkeId{0};
        uint32_t channelId{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_REMPOSTVAR_H
