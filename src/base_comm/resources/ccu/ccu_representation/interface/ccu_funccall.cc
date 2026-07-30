/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include "ccu_kernel.h"
#include "ccu_interface_assist_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

FuncCall::FuncCall(CcuRepContext *context, std::string label) : context(context), label(label)
{
    CcuInsGeneratorBase* insGenPtr = context->GetInsGenerator();
    Hccl::CHECK_NULLPTR(insGenPtr, "[FuncCall::FuncCall](label) insGenPtr is nullptr!");
    repFuncCall = std::make_shared<CcuRepFuncCall>(insGenPtr, label);
}

FuncCall::FuncCall(CcuRepContext *context, const Variable &funcAddr) : context(context)
{
    CcuInsGeneratorBase* insGenPtr = context->GetInsGenerator();
    Hccl::CHECK_NULLPTR(insGenPtr, "[FuncCall::FuncCall](funcAddr) insGenPtr is nullptr!");
    repFuncCall = std::make_shared<CcuRepFuncCall>(insGenPtr, funcAddr);
}

void FuncCall::AppendToContext()
{
    if (context == nullptr) {
        Hccl::THROW<Hccl::CcuApiException>("context is nullptr, func call, append to context");
    }
    return context->Append(repFuncCall);
}

}; // namespace CcuRep
}; // namespace hcomm