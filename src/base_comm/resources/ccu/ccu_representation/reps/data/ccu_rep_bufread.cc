/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "hcomm_c_adpt.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

CcuRepBufRead::CcuRepBufRead(CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, RemoteAddr src, CcuBuf dst, Variable len, CompletedEvent sem,
                             uint32_t mask)
    : insGenPtr(insGenPtr), channel(channel), src(src), dst(dst), len(len), sem(sem), mask(mask)
{
    type       = CcuRepType::BUF_READ;
    instrCount = insGenPtr->GetInstrCount(type);
}

bool CcuRepBufRead::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    insGenPtr->CcuRepBufReadTranslate(ccuKernel, instr, this, dep);
    instrId += instrCount;

    return translated;
}

std::string CcuRepBufRead::Describe()
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
    return Hccl::StringFormat("Read Rmt Mem[%u] To CcuBuf[%u], len[%u], ChannalId[%u], sem[%u], mask[%04x]",
                        src.addr.Id(), dst.Id(), len.Id(), channelImpl->GetChannelId(), sem.Id(), mask);
}

}; // namespace CcuRep
}; // namespace hcomm