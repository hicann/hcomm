# HcommLocalCopyOnThread

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-14T10:37:24.823Z pushedAt=2026-08-18T11:05:14.264Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Provides local memory copy. It copies the memory data of length len pointed to by src to the memory of the same length pointed to by dst.

## Function Prototype

```c
int32_t HcommLocalCopyOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t len)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the threads obtained by calling [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md).<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| dst | Output | Destination memory address, which is device memory. |
| src | Input | Source memory address, which is device memory. |
| len | Input | Data length (in bytes). |

## Return Value

int32_t: The API returns 0 on success and other values on failure.

## Constraints

dst and src are allocated device memory.

On Ascend 950PR/Ascend 950DT, this API can be called only on the device side in AICPU_TS mode.

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
// Perform D2D copy.
HcommLocalCopyOnThread(thread, outputMem, inputMem, memSize);
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
// Copy the parameters and launch the kernel.

// Device-side algorithm orchestration
uint64_t len = 256;
void *src = param.userIn;
void *dst = param.cclBuf;
// Execute the D2D copy.
HcommLocalCopyOnThread(thread, dst, src, len);
```
