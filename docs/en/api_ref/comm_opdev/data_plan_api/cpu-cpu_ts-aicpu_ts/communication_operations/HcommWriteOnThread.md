# HcommWriteOnThread

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T10:30:13.754Z pushedAt=2026-08-18T10:33:46.059Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Writes data to the specified memory on a channel, copying the memory data of length len from src to the memory area of the same length pointed to by dst. The API caller is the node where src resides. This API is asynchronous.

## Function Prototype

```c
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the threads obtained through [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md).<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channels obtained through [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md).<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| len | Input | Data length (in bytes). |

## Return Value

int32_t: The API returns 0 on success and other values on failure.

## Constraints

For Ascend 950PR/Ascend 950DT, only the UBC_TP, UBC_CTP, and UBoE communication protocols are supported.

## Example

```c
// Allocate communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS; // Used by Atlas A3 training products/Atlas A3 inference products
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS; // Used by Ascend 950PR/Ascend 950DT
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// Allocate communication channel resources.
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
uint32_t channelNum = 1;
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

// Write the local memory content to the remote memory.
HcommWriteOnThread(thread, channel, remoteBuffer, localBuffer, len);
```
