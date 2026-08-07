/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_interface_assist_v1.h"
#include "ccu_kernel.h"

#include "exception_util.h"
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

    void AppendToContext(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBase> rep)
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, AppendToContext assist[%d]", rep->Type());
        } else {
            return context->Append(rep);
        }
    }

    std::shared_ptr<CcuRep::CcuRepBlock> CurrentBlock(CcuRepContext* context)
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, currentBlock");
        }
        return context->CurrentBlock();
    }

    void SetCurrentBlock(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBlock> repBlock)
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, set currentBlock");
        }
        context->SetCurrentBlock(repBlock);
    }

    Variable CreateVariable(CcuRepContext* context)
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, CreateVar");
        }
        auto ctx = dynamic_cast<CcuKernel*>(context);
        if (ctx == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("Invalid context");
        }
        return ctx->CreateVariable();
    }

}; // namespace CcuRep
}; // namespace hcomm
