# HcommChannelDrainOnThread

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T10:17:49.708Z pushedAt=2026-08-18T07:49:36.267Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Not supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

On the AI CPU side, blocks and waits until all submitted read/write operations on the specified channel and thread complete naturally and the task queue is empty.

## Function Prototype

```c
int32_t HcommChannelDrainOnThread(ThreadHandle thread, ChannelHandle channel)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md).
| channel | Input | Communication channel handle, which is the channels obtained by calling [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md).<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

- When called on the AI CPU side, the communication engine is AICPU_TS.
- Only the RoCE communication protocol is supported.

## Example

```c
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
HcclComm comm;
ThreadHandle thread;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// Obtain local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// Obtain remote communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// Read the remote memory content to the local memory.
HcommReadOnThread(thread, channel, localBuffer, remoteBuffer, len);

HcommChannelDrainOnThread(thread, channel);
```
