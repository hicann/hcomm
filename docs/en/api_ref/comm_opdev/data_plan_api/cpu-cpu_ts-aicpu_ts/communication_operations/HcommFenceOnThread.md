# HcommFenceOnThread

<!-- md-trans-meta sourceCommit=7ff807bedd6173de4c7cb9ba16dadd5138b23868 translatedAt=2026-08-14T10:23:59.710Z pushedAt=2026-08-18T08:42:27.716Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Triggers a global data-plane fence operation on the host CPU side, flushes the currently initialized internal flush resources, and waits for completion.

This API is not bound to a specific communication channel. To wait for the completion of read and write operations that have been submitted on a specified communication channel, call [HcommChannelFenceOnThread](HcommChannelFenceOnThread.md) first.

## Function Prototype

```c
int32_t HcommFenceOnThread(ThreadHandle thread)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle. When called on the host CPU side, this parameter has no effect and can be set to 0.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

- For Ascend 950PR/Ascend 950DT, this API can be called only on the host CPU side.
- When called on the host CPU side, the communication engine is the CPU, the supported communication protocol is RoCE, and the `thread` parameter has no effect and can be set to 0.
- This API is used to flush the data plane on the host CPU side and does not wait for the read/write operations on the specified communication channel to complete. To wait for the read/write operations submitted on the channel to complete, call [HcommChannelFenceOnThread](HcommChannelFenceOnThread.md) first.
- If no internal resources need to be flushed, the API returns success.

## Example

### Collective Communication Example

```c
// Allocate communication channel resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
HcclComm comm;
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// Obtain the local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// Obtain the remote communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// Write the content of the local memory to the remote memory.
int32_t ret = HcommWriteNbiOnThread(0, channel, remoteBuffer, localBuffer, len);
ret = HcommChannelFenceOnThread(0, channel);

// Execute the data plane flush on the host CPU side and wait for completion.
ret = HcommFenceOnThread(0);
```
