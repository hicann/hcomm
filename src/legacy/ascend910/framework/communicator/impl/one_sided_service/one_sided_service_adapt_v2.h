/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_SERVICE_ADAPT_V2_H
#define HCCL_ONE_SIDED_SERVICE_ADAPT_V2_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult __attribute__((weak))
HcclRegisterMemV2(HcclComm comm, u32 remoteRank, int type, void* addr, u64 size, HcclMemDesc* desc);

HcclResult __attribute__((weak)) HcclDeregisterMemV2(HcclComm comm, HcclMemDesc* desc);

HcclResult __attribute__((weak)) HcclExchangeMemDescV2(
    HcclComm comm, u32 remoteRank, HcclMemDescs* local, int timeout, HcclMemDescs* remote, u32* actualNum);

HcclResult __attribute__((weak)) HcclEnableMemAccessV2(HcclComm comm, HcclMemDesc* remoteMemDesc, HcclMem* remoteMem);

HcclResult __attribute__((weak)) HcclDisableMemAccessV2(HcclComm comm, HcclMemDesc* remoteMemDesc);

HcclResult __attribute__((weak))
HcclBatchPutV2(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream);

HcclResult __attribute__((weak))
HcclBatchGetV2(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCCL_ONE_SIDED_SERVICE_ADAPT_V2_H
