/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_TEAM_H
#define HCCL_TEAM_H
#include <stdint.h>
#include <stddef.h>
#include "hcomm_res_defs.h"
#include "hcomm_team_defs.h"
#include "hcomm_team.h"
#include "hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
    CommAbiHeader header;
    const uint32_t *rankIds;     /* 用户希望填写的rankids */
    uint32_t  rankNum;
    uint32_t  selfRankId;
    uint32_t  netLayer;     /* 用户希望使用的网络层，0表示默认选择 */
    CommProtocol protocol;  /* 用户希望使用的通信协议，-1表示保留协议类型, 1个team仅支持一个协议 */
    HcommTeamSyncMemRequirement requirement;

    uint32_t reserved[4];
} HcclTeamCreateDesc;

typedef struct {
    CommAbiHeader   header;
    CommEngine      engine;          /* COMM_ENGINE_AICPU_TS / CCU 等 */
    uint32_t        notifyNum;       /* channel所需的notify数量 */
    CommProtocol    protocol;        /* 例如 COMM_PROTOCOL_HCCS / UBC_TP / UBOE */
    uint32_t        channelCnt;      /*  用户可以自行指定channel个数 */

    uint32_t        reserved[4];
} HcclTeamCreateChannelsDesc;

static const uint32_t HCCL_TEAM_CREATE_DESC_MAGIC_WORD = 0x0fcf0f14U;
static const uint32_t HCCL_TEAM_CREATE_DESC_VERSION = 1U;

static const uint32_t HCCL_TEAM_CREATE_CHANNELS_DESC_MAGIC_WORD = 0x0fcf0f15U;
static const uint32_t HCCL_TEAM_CREATE_CHANNELS_DESC_VERSION = 1U;

static inline HcclResult HcclTeamCreateDescInit(HcclTeamCreateDesc *desc)
{
    if (desc == NULL) {
        return HCCL_E_PTR;
    }

    (void)memset_s(desc, sizeof(HcclTeamCreateDesc), 0xFF, sizeof(HcclTeamCreateDesc));
    desc->header.version   = HCCL_TEAM_CREATE_DESC_VERSION;
    desc->header.magicWord = HCCL_TEAM_CREATE_DESC_MAGIC_WORD;
    desc->header.size      = sizeof(HcclTeamCreateDesc);
    desc->header.reserved  = 0;
    desc->rankIds     = NULL;
    desc->rankNum     = 0;
    desc->selfRankId  = 0;
    desc->netLayer    = 0;
    desc->protocol    = COMM_PROTOCOL_RESERVED;
    desc->requirement.signalCount  = 0;
    desc->requirement.counterCount = 0;
    desc->requirement.barrierCount = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        desc->reserved[i] = 0;
    }

    return HCCL_SUCCESS;
}

static inline HcclResult HcclTeamCreateChannelsDescInit(HcclTeamCreateChannelsDesc *desc)
{
    if (desc == NULL) {
        return HCCL_E_PTR;
    }

    (void)memset_s(desc, sizeof(HcclTeamCreateChannelsDesc), 0xFF, sizeof(HcclTeamCreateChannelsDesc));
    desc->header.version   = HCCL_TEAM_CREATE_CHANNELS_DESC_VERSION;
    desc->header.magicWord = HCCL_TEAM_CREATE_CHANNELS_DESC_MAGIC_WORD;
    desc->header.size      = sizeof(HcclTeamCreateChannelsDesc);
    desc->header.reserved  = 0;
    desc->engine     = COMM_ENGINE_RESERVED;
    desc->notifyNum  = 0;
    desc->protocol   = COMM_PROTOCOL_RESERVED;
    desc->channelCnt = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        desc->reserved[i] = 0;
    }

    return HCCL_SUCCESS;
}

/**
 * @brief Create a world team for HCCL communication.
 *
 * @param comm A pointer identifying the initialized communication resource.
 * @param worldTeam A pointer identifying the created world team handle.
 * @return HcclResult
 * @see HcclTeamDestroy()
 */
extern HcclResult HcclWorldTeamCreate(HcclComm comm, const HcclTeamCreateDesc *desc,
    HcommTeamHandle *worldTeam);

/**
 * @brief Create a sub team for HCCL communication.
 *
 * @param worldTeam A pointer identifying the world team handle.
 * @param desc A pointer identifying the sub team creation description.
 * @param team A pointer identifying the created sub team handle.
 * @return HcclResult
 * @see HcclTeamDestroy()
 */
extern HcclResult HcclSubTeamCreate(HcommTeamHandle worldTeam, const HcclTeamCreateDesc *desc,
                            HcommTeamHandle *team);

/**
 * @brief Destroy a team for HCCL communication.
 *
 * @param team A pointer identifying the team handle to be destroyed.
 * @return HcclResult
 * @see HcclWorldTeamCreate() / HcclSubTeamCreate()
 */
extern HcclResult HcclTeamDestroy(HcommTeamHandle team);

/**
 * @brief Register a memory window for HCCL communication.
 *
 * @param comm A pointer identifying the communication resource based on.
 * @param team A pointer identifying the team handle.
 * @param localMem A pointer identifying the user memory address.
 * @param window A pointer identifying the registered memory window handle.
 * @param flag The flag of this memory window, now only support 0
 * @return HcclResult
 * @see HcclTeamWindowDeregister()
 */
/* 要保证所有team上的rank都调用, 只能给put接口用, 入参team仅能为worldTeam */
/* HcclTeamWindowRegister生成的windows必须要调用HcclTeamChannelsCreate之后才能生效，如果只创建不调用无法使用该windows */
extern HcclResult HcclTeamWindowRegister(HcclComm comm, HcommTeamHandle worldTeam,
                                         const CommMem *localMem, HcommWindowHandle *window, HcommTeamWindowFlag flag);

/**
 * @brief Deregister a memory window for HCCL communication.
 *
 * @param team A pointer identifying the team handle.
 * @param window A pointer identifying the registered memory window handle.
 * @return HcclResult
 * @see HcclTeamWindowRegister()
 */
extern HcclResult HcclTeamWindowDeregister(HcommTeamHandle worldTeam, HcommWindowHandle window);

/**
 * @brief Create channels for a team for HCCL communication.
 *
 * @param   comm A pointer identifying the initialized communication resource.
 * @param   team A pointer identifying the team handle.
 * @param   desc A pointer identifying the team channel creation description.
 * @return  HcclResult
 */
extern HcclResult HcclTeamChannelsCreate(HcclComm comm, HcommTeamHandle team,
                                         const HcclTeamCreateChannelsDesc *desc);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_TEAM_H
