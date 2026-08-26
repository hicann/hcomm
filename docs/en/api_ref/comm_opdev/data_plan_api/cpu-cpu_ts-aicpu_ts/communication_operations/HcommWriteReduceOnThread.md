# HcommWriteReduceOnThread

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T10:32:57.134Z pushedAt=2026-08-18T10:37:42.369Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Writes data to the specified memory on the channel, performs the reduceOp operation on the memory data of length count*sizeof(dataType) in src and the memory data of the same length pointed to by dst, and outputs the result to dst. The API caller is the node where src resides.

## Function Prototype

```c
int32_t HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channels obtained through the [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) and [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) and [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| count | Input | Number of elements. |
| dataType | Input | Data type.<br>For the definition of the HcommDataType type, see [HcommDataType](../../../datatype_definition/HcommDataType.md).<br>For Ascend 950PR/Ascend 950DT, supported data types are as follows: int8, int16, int32, uint8, uint16, uint32, fp16, fp32, and bfp16.<br>For Atlas A3 training products/Atlas A3 inference products, supported data types are as follows: int8, int16, int32, float16, float32, and bfp16.<br>For Atlas A2 training products/Atlas A2 inference products, supported data types are as follows: int8, int16, int32, float16, float32, and bfp16. |
| reduceOp | Input | Reduction operation type. Supported values are as follows: sum, max, and min.<br>For the definition of the HcommReduceOp type, see [HcommReduceOp](../../../datatype_definition/HcommReduceOp.md). |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

None

## Example

```c
// Obtain communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS; // Used by Atlas A3 training products/Atlas A3 inference products
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS; // Used by Ascend 950PR/Ascend 950DT
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
uint64_t count = std::min(localBufferSize/sizeof(uint64_t), remoteBufferSize/sizeof(uint64_t));

// Reduce the local memory and remote memory data, and output the result to the remote memory.
HcommWriteReduceOnThread(thread, channel, remoteBuffer, localBuffer, count, HCOMM_DATA_TYPE_INT32, HCOMM_REDUCE_SUM);
```
