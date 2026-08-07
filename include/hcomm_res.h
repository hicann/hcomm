/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_H
#define HCOMM_RES_H

#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern HcommResult HcommEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle);

extern HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle);

extern HcommResult
HcommMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem, HcommMemHandle* memHandle);

extern HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle);

extern HcommResult
HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen);

extern HcommResult
HcommMemImport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen, CommMem* outMem);

extern HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen);

extern HcommResult HcommChannelCreate(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels);

/**
 * @brief 创建 Channel 配置对象
 * @param[out] config 输出的配置对象指针（不透明句柄）
 * @return HcommResult 执行结果状态码
 * @note 调用者获得 config 后通过 HcommChannelConfigSetInt/SetStr 设置属性，
 *       传入 HcommChannelCreateWithConfig 后由 HcommChannelConfigDestroy 销毁。
 */
extern HcommResult HcommChannelConfigCreate(HcommChannelConfig* config);

/**
 * @brief 销毁 Channel 配置对象
 * @param[in] config 配置对象指针
 * @return HcommResult 执行结果状态码
 */
extern HcommResult HcommChannelConfigDestroy(HcommChannelConfig config);

/**
 * @brief 设置 Channel 配置的整型属性
 * @param[in] config 配置对象指针
 * @param[in] type  属性类型，参见 HcommChannelConfigType
 * @param[in] value 属性值
 * @return HcommResult 执行结果状态码
 */
extern HcommResult HcommChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value);

/**
 * @brief 通过配置创建通信通道
 * @param[in] endpointHandle 网络设备句柄
 * @param[in] engine 通信引擎类型
 * @param[in] channelDescs 通道描述参数数组
 * @param[in] channelNum 通道数量
 * @param[in] config Channel 配置对象指针（可为 NULL，等价于 HcommChannelCreate）
 * @param[out] channels 输出的通道句柄数组
 * @return HcommResult 执行结果状态码
 * @note 当 config 中 IS_SHARED_QUEUE=true 时：
 *       - 仅支持 UB 网络语义协议（UBC_CTP/UBC_TP），不支持 UBMem/RoCE/UBOE/UBG。
 *       - 使用相同 endpointHandle 多次调用本接口创建的 Channel 共享同一个 Jetty。
 *       - 使用不同 endpointHandle 调用创建的 Channel 不共享 Jetty。
 *       - 共享 Jetty 的不同 Channel 不支持并发使用，需由调用者按业务顺序串行调用。
 *       - 销毁 endpointHandle 前需确保所有共享 Jetty 的 Channel 已销毁。
 */
extern HcommResult HcommChannelCreateWithConfig(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    HcommChannelConfig config, ChannelHandle* channels);

extern HcommResult HcommChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList);

extern HcommResult HcommChannelDestroy(const ChannelHandle* channels, uint32_t channelNum);

extern HcommResult
HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t* notifyNumPerThread, ThreadHandle* threads);

extern HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum);

extern HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port);

extern HcommResult
HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc* endpointDesc, bool* value);

extern HcommResult HcommMemAlloc(void** ptr, size_t size);

extern HcommResult HcommMemFree(void* ptr);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOMM_RES_H
