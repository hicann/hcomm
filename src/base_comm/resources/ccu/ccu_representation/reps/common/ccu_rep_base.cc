/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"

namespace hcomm {
namespace CcuRep {

    CcuRepBase::CcuRepBase() {}

    CcuRepBase::~CcuRepBase() {}

    CcuRepType CcuRepBase::Type() const { return type; }

    bool CcuRepBase::Translated() const { return translated; }

    uint16_t CcuRepBase::StartInstrId() const { return instrId; }
    uint16_t CcuRepBase::InstrCount() { return instrCount; }

}; // namespace CcuRep
}; // namespace hcomm
