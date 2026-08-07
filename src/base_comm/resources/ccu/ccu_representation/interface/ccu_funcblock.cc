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
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

    FuncBlock::FuncBlock(CcuRepContext* context, std::string label, uint16_t callLayer)
        : context(context),
          label(label),
          callLayer(callLayer)
    {}

    FuncBlock::~FuncBlock() {}

}; // namespace CcuRep
}; // namespace hcomm
