# HcclCommUnsetMemoryRange

<!-- md-trans-meta sourceCommit=ffc5b1340e1ba3599f61afdda8ea78f73c8e36d1 translatedAt=2026-08-14T08:41:38.568Z pushedAt=2026-08-15T03:49:24.815Z -->

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

Notifies the HCCL communicator to stop using the reserved virtual memory.

## Function Prototype

```c
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void *baseVirPtr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. It is recommended to use the largest communicator in the server, that is, the communicator covering the maximum number of devices. |
| baseVirPtr | Input | Base address of the reserved virtual memory.<br>The specified base address must have been successfully set by [HcclCommSetMemoryRange](HcclCommSetMemoryRange.md); otherwise, an error is reported. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

If active memory still exists in this virtual address space, this API fails to execute.
