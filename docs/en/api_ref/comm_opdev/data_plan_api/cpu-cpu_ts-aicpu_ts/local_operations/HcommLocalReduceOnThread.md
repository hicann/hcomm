# HcommLocalReduceOnThread

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T10:37:45.798Z pushedAt=2026-08-18T11:07:50.720Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Provides a local reduce operation that performs the reduceOp operation on the memory data of length count\*sizeof\(dataType\) pointed to by src and the memory data of the same length pointed to by dst, and outputs the result to dst.

## Function Prototype

```c
int32_t HcommLocalReduceOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| dst | Output | Destination address, device memory. |
| src | Input | Source address, device memory. |
| count | Input | Number of elements. |
| dataType | Input | Data type. Supported types: int8, int16, int32, float16, float32, bfp16.<br>For the definition of the HcommDataType type, see [HcommDataType](../../../datatype_definition/HcommDataType.md). |
| reduceOp | Input | Reduce operation type. Supported types: sum, max, min.<br>For the definition of the HcommReduceOp type, see [HcommReduceOp](../../../datatype_definition/HcommReduceOp.md). |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

The dst and src memory must be device memory.

On Ascend 950PR/Ascend 950DT, this API can be called only in AICPU_TS mode and on the device side.

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream stream;
aclrtCreateStream(&stream);
ThreadHandle thread;
HcclResult result = HcclThreadAcquireWithStream(comm, engine, stream, 2, &thread);

// Allocate device memory.
uint64_t memSize = 256;
s32 policy = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMallocAttrValue moduleIdValue;
moduleIdValue.moduleId = HCCL;
aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

void* inputMem;
void* outputMem;
aclrtMallocWithCfg(&inputMem, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
aclrtMallocWithCfg(&outputMem, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
// Execute the reduce operation on the device side.
uint64_t count = memSize / SIZE_TABLE[HCOMM_DATA_TYPE_FP32];
HcommLocalReduceOnThread(thread, outputMem, inputMem, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
```

On Ascend 950PR/Ascend 950DT, this function must be compiled for use on the device side:

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;

void *cclBufferAddr = nullptr;
uint64_t cclBufferSize = 0;
HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize);
ThreadHandle thread;
HcclThreadAcquire(comm, engine, 1, 1, &thread);

// Allocate other resources.
// Copy parameters and launch the kernel.

// Device-side algorithm orchestration
uint64_t len = 256;
void *src = param.userIn;
void *dst = param.cclBuf;
uint64_t sizeOfFP32 = 4;
uint64_t count = len / sizeOfFP32;
HcommLocalReduceOnThread(thread, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
```
