/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_NIC_PLUGIN_DISPATCHER_H
#define HCOMM_NIC_PLUGIN_DISPATCHER_H

#include "hcomm_c_adpt.h"
#include "hcomm_primitives.h"

namespace hcomm {

bool IsPluginEndpoint(EndpointHandle handle);
bool IsPluginChannel(ChannelHandle handle);

HcommResult PluginEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle, bool &handled);
HcommResult PluginEndpointGet(EndpointHandle handle, void **endpoint, bool &handled);
HcommResult PluginEndpointDestroy(EndpointHandle handle, bool &handled);

HcommResult PluginMemReg(EndpointHandle handle, const char *memTag,
    const CommMem *mem, HcommMemHandle *memHandle, bool &handled);
HcommResult PluginMemUnreg(EndpointHandle handle, HcommMemHandle memHandle, bool &handled);
HcommResult PluginMemExport(EndpointHandle handle, HcommMemHandle memHandle,
    void **memDesc, uint32_t *memDescLen, bool &handled);
HcommResult PluginMemImport(EndpointHandle handle, const void *memDesc, uint32_t descLen,
    CommMem *outMem, bool &handled);
HcommResult PluginMemUnimport(EndpointHandle handle, const void *memDesc, uint32_t descLen, bool &handled);

HcommResult PluginChannelCreate(EndpointHandle endpointHandle, const HcommChannelDesc *channelDesc,
    ChannelHandle *channelHandle);
HcommResult PluginChannelCreate(EndpointHandle endpointHandle, CommEngine engine,
    const HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels, bool &handled);
HcommResult PluginChannelGet(ChannelHandle handle, void **channel, bool &handled);
HcommResult PluginChannelGetStatus(ChannelHandle handle, int32_t *status, bool &handled);
HcommResult PluginChannelGetNotifyNum(ChannelHandle handle, uint32_t *notifyNum, bool &handled);
HcommResult PluginChannelDestroy(ChannelHandle handle, bool &handled);
HcommResult PluginChannelUpdateMemInfo(ChannelHandle handle, HcommMemHandle *memHandles, uint32_t memHandleNum,
    bool &handled);
HcommResult PluginChannelGetRemoteMems(ChannelHandle handle, uint32_t *memNum, CommMem **remoteMem,
    char ***memInfos, bool &handled);

HcommResult PluginChannelWrite(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src, uint64_t len,
    bool &handled);
HcommResult PluginChannelBatchTransfer(ChannelHandle handle, ThreadHandle thread,
    const HcommBatchTransferDesc *transferDescs, uint32_t transferDescNum, bool &handled);
HcommResult PluginChannelWriteReduce(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, bool &handled);
HcommResult PluginChannelWriteWithNotify(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx, bool &handled);
HcommResult PluginChannelWriteReduceWithNotify(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, uint32_t remoteNotifyIdx, bool &handled);
HcommResult PluginChannelRead(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src, uint64_t len,
    bool &handled);
HcommResult PluginChannelReadReduce(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, bool &handled);
HcommResult PluginChannelWriteNbi(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src, uint64_t len,
    bool &handled);
HcommResult PluginChannelWriteWithNotifyNbi(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx, bool &handled);
HcommResult PluginChannelReadNbi(ChannelHandle handle, ThreadHandle thread, void *dst, const void *src, uint64_t len,
    bool &handled);
HcommResult PluginChannelNotifyRecord(ChannelHandle handle, ThreadHandle thread, uint32_t remoteNotifyIdx,
    bool &handled);
HcommResult PluginChannelNotifyWait(ChannelHandle handle, ThreadHandle thread, uint32_t localNotifyIdx,
    uint32_t timeOut, bool &handled);
HcommResult PluginChannelFence(ChannelHandle handle, ThreadHandle thread, bool &handled);

} // namespace hcomm

#endif // HCOMM_NIC_PLUGIN_DISPATCHER_H
