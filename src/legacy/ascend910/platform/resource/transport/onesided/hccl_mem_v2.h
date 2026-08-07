/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_MEM_V2_H
#define HCCL_MEM_V2_H

#include "hccl_types.h"
#include "hccl_mem_defs.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult __attribute__((weak)) HcclMemRegV2(HcclNetDev netDev, const HcclMem* mem, HcclBuf* buf);
HcclResult __attribute__((weak)) HcclMemDeregV2(const HcclBuf* buf);
HcclResult __attribute__((weak)) HcclMemExportV2(HcclBuf* buf, char** outDesc, uint64_t* outDescLen);
HcclResult __attribute__((weak))
HcclMemImportV2(const char* description, uint64_t descLen, bool isRemote, HcclBuf* outBuf, HcclNetDev netDev);
HcclResult __attribute__((weak)) HcclMemCloseV2(HcclBuf* buf);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCCL_MEM_V2_H
