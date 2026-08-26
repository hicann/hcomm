# HcommWriteWithNotifyOnThread

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T10:34:48.036Z pushedAt=2026-08-18T10:56:24.280Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Writes data to the specified memory on the channel, copies the memory data of length len from src to the memory area of the same length pointed to by dst, and sends a synchronization signal to the node where dst resides. The API caller is the node where src resides. This API is asynchronous.

## Function Prototype

```c
int32_t HcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len, uint32_t remoteNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the threads obtained through [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md).<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channels obtained through [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md).<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| len | Input | Data length (in bytes). |
| remoteNotifyIdx | Input | Notify index of the other end of the communication channel.<br>Value range: [0, notifyNum in the channelDescs parameter passed to [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)). |

## Return Value

int32_t: The API returns 0 on success and other values on failure.

## Constraints

This API must be used together with [HcommChannelNotifyWaitOnThread](HcommChannelNotifyWaitOnThread.md).

On Ascend 950PR/Ascend 950DT, this API can be called only in AICPU_TS mode on the device side.

## Example

```c
// Allocate communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
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

// For Ascend 950PR/Ascend 950DT, call the following APIs on the device side.

// Write the content of the local memory to the peer memory and notify the remote end.
uint32_t rmtNotifyIdx = 0;
HcommWriteWithNotifyOnThread(thread, channel, remoteBuffer, localBuffer, len, rmtNotifyIdx);

// Perform data plane operations.
// ...

// Wait for the remote end to notify the local end.
uint32_t lclNotifyIdx = 0;
uint32_t notifyTimeout = 0;
HcommChannelNotifyWaitOnThread(thread, channel, lclNotifyIdx, notifyTimeout);
```
