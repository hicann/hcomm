/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rma_connection.h"

namespace Hccl {

RmaConnection::RmaConnection(Socket* socket, const RmaConnType rmaConnType) : socket(socket), rmaConnType(rmaConnType)
{
    status = RmaConnStatus::INIT;
}

RmaConnection::~RmaConnection()
{
    if (status != RmaConnStatus::CLOSE) {
        remoteBufs.clear();
        status = RmaConnStatus::CLOSE;
    }
}

void RmaConnection::Close()
{
    remoteBufs.clear();
    status = RmaConnStatus::CLOSE;
}

RmaConnStatus RmaConnection::GetStatus() { return status; }

void RmaConnection::Bind(RemoteRmaBuffer* remoteRmaBuf, BufferType bufType)
{
    HCCL_INFO(
        "[RmaConnection][%s] bind bufType[%s] Buffer[%s].", Describe().c_str(), bufType.Describe().c_str(),
        remoteRmaBuf->Describe().c_str());
    remoteBufs[bufType] = remoteRmaBuf;
}

RemoteRmaBuffer* RmaConnection::GetRemoteRmaBuffer(const BufferType& bufType)
{
    auto iter = remoteBufs.find(bufType);
    if (iter != remoteBufs.end()) {
        return remoteBufs[bufType];
    } else {
        return nullptr;
    }
}

unique_ptr<BaseTask>
RmaConnection::PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config)
{
    MACRO_THROW(NotSupportException, StringFormat("RmaConnection not support this function."));
}

unique_ptr<BaseTask> RmaConnection::PrepareReadReduce(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
    const SqeConfig& config)
{
    MACRO_THROW(NotSupportException, StringFormat("RmaConnection not support this function."));
}

unique_ptr<BaseTask>
RmaConnection::PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config)
{
    MACRO_THROW(NotSupportException, StringFormat("RmaConnection not support this function."));
}

unique_ptr<BaseTask> RmaConnection::PrepareWriteReduce(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
    const SqeConfig& config)
{
    MACRO_THROW(NotSupportException, StringFormat("RmaConnection not support this function."));
}

unique_ptr<BaseTask>
RmaConnection::PrepareInlineWrite(const MemoryBuffer& remoteMemBuf, u64 data, const SqeConfig& config)
{
    MACRO_THROW(NotSupportException, StringFormat("RmaConnection not support this function."));
}

unique_ptr<BaseTask> RmaConnection::PrepareWriteWithNotify(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, u64 data, const MemoryBuffer& remoteNotifyMemBuf,
    const SqeConfig& config)
{
    MACRO_THROW(NotSupportException, StringFormat("RmaConnection not support this function."));
}

unique_ptr<BaseTask> RmaConnection::PrepareWriteReduceWithNotify(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp, u64 data,
    const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config)
{
    MACRO_THROW(NotSupportException, StringFormat("RmaConnection not support this function."));
}

} // namespace Hccl
