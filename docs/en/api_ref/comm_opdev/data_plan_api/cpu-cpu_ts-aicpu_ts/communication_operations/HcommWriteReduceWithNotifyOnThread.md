# HcommWriteReduceWithNotifyOnThread

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T10:33:29.862Z pushedAt=2026-08-18T10:41:52.923Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Writes data to the specified memory on a channel, performs the reduceOp operation on the memory data of length count\*sizeof\(dataType\) in src and the memory data of the same length pointed to by dst, outputs the result to dst, and sends a synchronization signal to the node where dst resides. The API caller is the node where src resides. This API is asynchronous.

## Function Prototype

```c
int32_t HcommWriteReduceWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channels obtained through the [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which uses the HCCL communication memory of the remote end of the specified channel. |
| src | Input | Source memory address, which uses the HCCL communication memory of the local rank in the communicator. |
| count | Input | Number of elements. |
| dataType | Input | Data type.<br>For the definition of the HcommDataType type, see [HcommDataType](../../../datatype_definition/HcommDataType.md).<br>For Ascend 950PR/Ascend 950DT, the supported data types are int8, int16, int32, uint8, uint16, uint32, float16, float32, and bfp16. |
| reduceOp | Input | Reduction operation type. Supported values: sum, max, and min.<br>For the definition of the HcommReduceOp type, see [HcommReduceOp](../../../datatype_definition/HcommReduceOp.md). |
| remoteNotifyIdx | Input | Notify index of the other end of the communication channel.<br>Value range: [0, notifyNum in the channelDescs parameter passed to the [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API). |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

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

// Copy parameters and launch the kernel.
// Device-side algorithm orchestration
uint64_t len = std::min(localBufferSize, remoteBufferSize);
uint64_t sizeOfFP32 = 4;
uint64_t count = len / sizeOfFP32;

// Write the local memory content to the remote memory and notify the remote end.
uint32_t rmtNotifyIdx = 0;
HcommWriteReduceWithNotifyOnThread(thread, channel, remoteBuffer, localBuffer, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM, rmtNotifyIdx);

// Data plane operations
// ...

// Wait for the remote end to notify the local end.
uint32_t lclNotifyIdx = 0;
uint32_t notifyTimeout = 0;
HcommChannelNotifyWaitOnThread(thread, channel, lclNotifyIdx, notifyTimeout);
```
