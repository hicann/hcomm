/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-05-21
 */

#ifndef HCOMM_CCU_REPRESENTATION_REM_MEM_H
#define HCOMM_CCU_REPRESENTATION_REM_MEM_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepRemMem : public CcuRepBase {

public:
    CcuRepRemMem(CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, RemoteAddr rem);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;

    ChannelHandle GetChannel() { return channel; }
    RemoteAddr GetRem() { return rem; }
private:
    CcuInsGeneratorBase* insGenPtr{nullptr};
    ChannelHandle channel;

    RemoteAddr rem{};

};

};     // namespace CcuRep
};     // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_REM_MEM_H