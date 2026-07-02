/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "COMM_STUB"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "sim_log.h"
#include "hccp_common.h"
#include "sim_ip_address.h"
#include "runtime/base.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_ops.h"


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
extern HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);
extern HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm);
extern HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm);
extern HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm);
extern HcclResult HcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank,
    HcclCommConfig *config, HcclComm *comm);

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo)
{
    (void) rootInfo;
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm)
{
    (void) nRanks;
    (void) rootInfo;
    // 解析ranktable文件，初始化通信域相关数据库表项
    const char* clusterInfo = std::getenv("RANK_TABLE_FILE");
    if (!clusterInfo) {
        HCCL_VM_ERROR("RANK_TABLE_FILE env not set, please check your config.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclCommInitClusterInfo(clusterInfo, rank, comm);
}

HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm)
{
    (void) nRanks;
    (void) rootInfo;
    const char* clusterInfo = getenv("RANK_TABLE_FILE");
    HcclCommConfig *cfg = const_cast<HcclCommConfig *>(config);
    return HcclCommInitClusterInfoConfig(clusterInfo, rank, cfg, comm);
}

#ifdef __cplusplus
}
#endif  // __cplusplus
