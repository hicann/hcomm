/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef CCU_RES_H
#define CCU_RES_H

#include "ccu_res_defs.h"
#include "ccu_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 创建 CCU 实例资源描述符
 * @param[in]  dieId    IO Die 编号，取值范围 [0, CCU_MAX_IODIE_NUM)
 * @param[out] resDesc  输出指针，成功时写入 HcommCcuResDescHandle 句柄；调用方负责通过
 *                      HcommCcuInsResDescDestroy 释放
 * @return CCU_SUCCESS  创建成功
 *         CCU_E_PARA   dieId 超出合法范围
 *         CCU_E_PTR    resDesc 为 nullptr
 *         CCU_E_INTERNAL 内部资源描述符创建或登记失败
 */
extern CcuResult HcommCcuInsResDescCreate(uint32_t dieId, HcommCcuResDescHandle *resDesc);

/**
 * @brief 销毁 CCU 实例资源描述符
 * @param[in] resDesc 资源描述符句柄，由 HcommCcuInsResDescCreate 创建
 * @return CCU_SUCCESS  销毁成功
 *         CCU_E_NOT_FOUND resDesc 未注册
 */
extern CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle resDesc);

/**
 * @brief 查询 CCU 资源描述符中已设置的 Die ID
 * @param[in]  resDesc  资源描述符句柄
 * @param[out] dieId    输出指针，接收已设置的 Die ID
 * @return CCU_SUCCESS  查询成功
 *         CCU_E_PTR    dieId 为 nullptr
 *         CCU_E_NOT_FOUND resDesc 未注册
 */
extern CcuResult HcommCcuInsResDescQueryDieId(HcommCcuResDescHandle resDesc, uint32_t *dieId);

/**
 * @brief 按资源类型设置 CCU 资源描述符中的资源数量
 * @param[in,out] resDesc  由 HcommCcuInsResDescCreate 创建的资源描述符句柄，成功时更新指定资源类型的资源数量
 * @param[in] resType  资源类型，取 HcommCcuResType 枚举值
 * @param[in] resNum   期望的资源数量；设置为 0 表示不申请该类型资源
 * @return CCU_SUCCESS  设置成功
 *         CCU_E_PARA   resType 不是合法的 HcommCcuResType 枚举值
 *         CCU_E_NOT_FOUND resDesc 未注册
 */
extern CcuResult HcommCcuInsResDescSetNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t resNum);

/**
 * @brief 按资源类型查询 CCU 资源描述符中已设置的资源数量
 * @param[in]  resDesc  资源描述符句柄
 * @param[in]  resType  资源类型，取 HcommCcuResType 枚举值
 * @param[out] resNum   输出指针，接收已设置的资源数量
 * @return CCU_SUCCESS  查询成功
 *         CCU_E_PARA   resType 超出合法范围
 *         CCU_E_PTR    resNum 为 nullptr
 *         CCU_E_NOT_FOUND resDesc 未注册
 */
extern CcuResult HcommCcuInsResDescQueryNum(
    HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t *resNum);

/**
 * @brief 基于资源描述符创建 CCU 实例。
 * @param[in] resDescs 资源描述符句柄数组。
 * @param[in] resDescNum 资源描述符数量。
 * @param[out] ccuInsHandle 创建成功后返回的 CCU 实例句柄，不可为 nullptr。
 * @return CCU_SUCCESS  创建成功
 *         CCU_E_PTR    resDescs 或 ccuInsHandle 为 nullptr，或内部资源包分配失败
 *         CCU_E_PARA   resDescNum 超出合法范围，或资源描述符中的 ioDie ID 重复
 *         CCU_E_INTERNAL 内部 CCU 实例创建或登记失败
 */
extern CcuResult HcommCcuInsCreate(
    const HcommCcuResDescHandle *resDescs, uint32_t resDescNum, CcuInsHandle *ccuInsHandle);

/**
 * @brief 使用当前 Device 上所有已使能 ioDie 的全部资源创建 CCU 实例。
 * @param[in] dieIds 保留参数，当前版本不读取，调用方应传入 nullptr。
 * @param[in] dieNum 保留参数，当前版本不读取，调用方应传入 0。
 * @param[out] ccuInsHandle 创建成功后返回的 CCU 实例句柄，不可为 nullptr。
 * @return CCU_SUCCESS  创建成功
 *         CCU_E_PTR    ccuInsHandle 为 nullptr，或内部资源包分配失败
 *         CCU_E_INTERNAL 内部 CCU 实例创建或登记失败
 */
extern CcuResult HcommCcuInsCreateDefault(
    const uint32_t *dieIds, uint32_t dieNum, CcuInsHandle *ccuInsHandle);

/**
 * @brief 销毁 CCU 实例。
 * @param[in] ccuInsHandle CCU 实例句柄。
 * @return CcuResult。
 */
extern CcuResult HcommCcuInsDestroy(CcuInsHandle ccuInsHandle);

/**
 * @brief 查询 CCU 实例在资源描述符所属 ioDie 上占用的资源。
 * @param[in] ccuInsHandle CCU 实例句柄。
 * @param[in,out] resDesc 由 HcommCcuInsResDescCreate 创建的资源描述符句柄；接口读取创建 resDesc 时设置的
 *                        dieId 作为待查询 ioDie，查询成功后写入该 ioDie 上的占用资源数量。
 * @return CcuResult。
 */
extern CcuResult HcommCcuInsQueryResDesc(CcuInsHandle ccuInsHandle, HcommCcuResDescHandle resDesc);

/**
 * @brief 查询指定 Die 上的剩余 CCU 资源（查询最大连续资源）
 * @param[in,out] resDesc 由 HcommCcuInsResDescCreate 创建的资源描述符句柄；接口读取创建 resDesc 时设置的
 *                        dieId 作为待查询 ioDie，查询成功后写入各资源类型的最大连续剩余数量
 * @return CCU_SUCCESS     查询成功
 *         CCU_E_NOT_FOUND resDesc 未注册
 *         CCU_E_PARA      dieId 超出合法范围
 *         CCU_E_UNAVAIL   指定的 Die 未使能
 *         CCU_E_INTERNAL  底层资源查询失败
 */
extern CcuResult HcommCcuQueryRemainResDesc(HcommCcuResDescHandle resDesc);

/**
 * @brief 查询 CCU Kernel 的资源诉求。
 * @param[in] kernelFunc CCU Kernel 函数指针，不可为 nullptr。
 * @param[in] kernelArgs CCU Kernel 参数数组；argNum 为 0 时可为 nullptr。
 * @param[in] argNum CCU Kernel 参数数量。
 * @param[in,out] resDesc 由 HcommCcuInsResDescCreate 创建的资源描述符句柄；接口读取创建 resDesc 时设置的
 *                        dieId 作为资源统计的目标 ioDie，查询成功后写入资源诉求。
 * @return CcuResult。
 */
extern CcuResult HcommCcuKernelQueryResReq(
    const void *kernelFunc, const void **kernelArgs, uint32_t argNum, HcommCcuResDescHandle resDesc);

/**
 * @brief 查询/计算一段本端内存区域的 CCU 访问 token，供 host 端组装 LocalAddr/RemoteAddr 使用。
 * @param[in]  srcVa     内存区域起始虚地址（VA），必须为已注册的合法地址，不可为 0。
 * @param[in]  size      内存区域长度，单位：字节，必须 > 0 且不超过该 VA 对应映射的实际大小。
 * @param[out] tokenInfo 输出指针，指向单个 uint64_t（非数组），成功时写入计算得到的 token 值；不可为 nullptr。
 * @return CcuResult。CCU_SUCCESS 表示成功；CCU_E_PARA 表示 srcVa/size 为 0 或非法；
 *         CCU_E_PTR 表示 tokenInfo 为 nullptr；其余为底层驱动错误码。
 * @note 仅可在 host 端调用，不能在 kernel 函数体内调用。token 属安全信息，调用方不应打印；
 *       其生命周期与对应内存注册绑定。跨 rank 场景下本端 token 需经带外通道交换给对端组装 RemoteAddr。
 */
extern CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t *tokenInfo);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CCU_RES_H
