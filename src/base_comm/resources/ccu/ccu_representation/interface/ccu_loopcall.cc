/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_loopcall_v1.h"

#include "exception_util.h"
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

LoopCall::LoopCall(CcuRepContext *context, const std::string &label) : context(context), label(label)
{
    // repLoopCall = std::make_shared<CcuRepLoopCall>(label);
}

void LoopCall::AppendToContext()
{
    if (context == nullptr) {
        Hccl::THROW<Hccl::CcuApiException>("context is nullptr, loopCall");
    }
    return context->Append(repLoopCall);
}

}; // namespace CcuRep
}; // namespace hcomm