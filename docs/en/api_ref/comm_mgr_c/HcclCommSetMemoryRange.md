# HcclCommSetMemoryRange

<!-- md-trans-meta sourceCommit=ffc5b1340e1ba3599f61afdda8ea78f73c8e36d1 translatedAt=2026-08-14T08:37:48.928Z pushedAt=2026-08-14T10:04:33.137Z -->

> [!NOTE] Note
> This API is for trial use and may be changed later. It cannot be used in production environments.

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

After a user successfully reserves virtual memory by calling the aclrtReserveMemAddress API, the user can call this API to notify HCCL of the reserved virtual memory address. After this API is called, the virtual memory is visible to all communicators in the current process.

## Function Prototype

```c
HcclResult HcclCommSetMemoryRange(HcclComm comm, void *baseVirPtr, size_t size, size_t alignment, uint64_t flags)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. You are advised to use the largest communicator within a server, that is, the communicator covering the maximum number of devices. |
| baseVirPtr | Input | Base address of the virtual memory to be reserved, that is, the virtual memory address output by the aclrtReserveMemAddress API. |
| size | Input | Size of the virtual memory, in bytes. |
| alignment | Input | Reserved field.<br>Currently, only "0" is supported. |
| flags | Input | Reserved field.<br>Currently, only "0" is supported. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- When this API is called for the first time within a communicator, a link establishment operation is performed. Therefore, when calling this API for the first time, ensure that all processes in the communicator call this API at the same time to avoid link establishment timeout. This constraint does not apply to subsequent calls.
- This API can be called only within a communicator whose scope is a single server. Otherwise, an error is reported.
- When this API is called multiple times, the input memory addresses must not be duplicated or have overlapping ranges.
- For other constraints, see [General Constraints](./zero_copy_readme.md).
