/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "hcomm_c_adpt.h"
#include "ccu_ins_generater_v1.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_ins_generater_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

CcuRepBufWrite::CcuRepBufWrite(CcuInsGeneraterBase* insGenPtr, const ChannelHandle channel, CcuBuf src, RemoteAddr dst, Variable len, CompletedEvent sem,
                               uint16_t mask)
    : insGenPtr(insGenPtr), channel(channel), src(src), dst(dst), len(len), sem(sem), mask(mask)
{
    type       = CcuRepType::BUF_WRITE;
    instrCount = 1;
}

bool CcuRepBufWrite::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    instrCount = insGenPtr->GetInstrCount(type);
    insGenPtr->CcuRepBufWriteTranslate(ccuKernel, instr, this, dep);
    instrId += instrCount;

    return translated;
}

std::string CcuRepBufWrite::Describe()
{
    void *channelPtr{nullptr};
    auto ret = HcommChannelGet(channel, &channelPtr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("failed to get ccu channel, type[%d]", type);
    }

    auto *channelImpl = dynamic_cast<CcuUrmaChannel *>(static_cast<Channel *>(channelPtr));
    if (channelImpl == nullptr) {
        Hccl::THROW<Hccl::CcuApiException>("[%s] failed to cast channel[0x%llx] to CcuUrmaChannel",
            __func__, channel);
    }
    return Hccl::StringFormat("Write CcuBuf[%u] To Rmt Mem[%u], len[%u], ChannalId[%u], sem[%u], mask[%04x]", src.Id(),
                        dst.addr.Id(), len.Id(), channelImpl->GetChannelId(), sem.Id(), mask);
}

}; // namespace CcuRep
}; // namespace hcomm