/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Description: ccu executor -- load sqe arg to xn
 * Author: caiyifan
 */

#include "load_sqe_args_to_xn_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册LoadSqeArgsToXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC_V1(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADSQEARGSTOXN_CODE, LoadSqeArgsToXnExecutor);

void LoadSqeArgsToXnExecutor::Parser() {
    if (version_ == RunnerCcuVersion::CCU_V1) {
        xnId_     = instr_.v1.loadSqeArgsToXn.xnId;
        sqeArgId_ = instr_.v1.loadSqeArgsToXn.sqeArgsId;
    } else {
        HCCL_VM_ERROR("Unsupported CCU version: {}", RunnerCcuVersionToString(version_));
    }
}

void LoadSqeArgsToXnExecutor::Run() {
    // 加载sqe参数至xn寄存器，只需更新对应dev的ccu资源映射表即可
    auto &ccuResMgr = CcuResourceManager::GetInstance();
    uint64_t sqeArgValue = ccuResMgr.GetSqeArgValue(rankId_, dieId_, sqeArgId_);

    HCCL_VM_DEBUG("Load arg: locCcu[{}:{}], XnId=[{}], argId=[{}], value=[{}]",
        rankId_, dieId_, xnId_, sqeArgId_, sqeArgValue);
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xnId_, sqeArgValue);
}

std::string LoadSqeArgsToXnExecutor::Describe() {
    uint64_t sqeArgValue = CcuResourceManager::GetInstance().GetSqeArgValue(rankId_, dieId_, sqeArgId_);
    return HcclSim::StringFormat("[Simulation Execute] locCcu[%d:%d], Load SqeArg[%u]-Value[%x] to Xn[%u]\n", rankId_, dieId_, sqeArgId_, sqeArgValue, xnId_);
}
