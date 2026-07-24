/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2025-05-21
 */

#include "ccu_rep_v1.h"
#include "ccu_assist_v1.h"

#include "string_util.h"

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "hcomm_c_adpt.h"

#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_ins_generater_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

CcuRepRemMem::CcuRepRemMem(CcuInsGeneraterBase* insGenPtr, const ChannelHandle channel, RemoteAddr rem)
    : insGenPtr(insGenPtr), channel(channel), rem(rem) 
{
    type = CcuRepType::REM_MEM;
    instrCount = 2;  // 指令数为2个
}

bool CcuRepRemMem::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    instrCount = insGenPtr->GetInstrCount(type);
    insGenPtr->CcuRepRemMemTranslate(ccuKernel, instr, this);
    instrId += instrCount;

    return translated;
}

std::string CcuRepRemMem::Describe()
{
    return Hccl::StringFormat("Get Remote Buffer Addr and TokenInfo By Transport");
}

}; // namespace CcuRep
}; // namespace hcomm