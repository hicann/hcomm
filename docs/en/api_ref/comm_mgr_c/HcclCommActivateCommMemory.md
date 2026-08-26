# HcclCommActivateCommMemory

<!-- md-trans-meta sourceCommit=ffc5b1340e1ba3599f61afdda8ea78f73c8e36d1 translatedAt=2026-08-14T08:28:26.022Z pushedAt=2026-08-14T09:09:00.256Z -->

> [!NOTE] Note
>
> This API is for trial use and may be changed in the future. It cannot be used in production environments.

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

Activates the reserved virtual memory. Only when the activated memory is used as the input and output of communication operators can the zero-copy feature be enabled.

## Function Prototype

```c
HcclResult HcclCommActivateCommMemory(HcclComm comm, void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. You are advised to use the largest communicator in the server, that is, the communicator covering the maximum number of devices. |
| virPtr | Input | Virtual memory address to be activated, that is, the to-be-mapped virtual memory address passed in when the user calls the aclrtMapMem API to map physical memory to virtual memory. |
| size | Input | Size of the memory to be activated, in bytes. |
| offset | Input | Reserved field.<br>Currently, only "0" is supported. |
| handle | Input | Handle of the allocated physical memory information, that is, the handle of the device physical memory information allocated by the user by calling the aclrtMallocPhysical API. |
| flags | Input | Reserved field.<br>Currently, only "0" is supported. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The virtual memory address to be activated must be within the address range set by [HcclCommSetMemoryRange](HcclCommSetMemoryRange.md).
- The virtual memory address must not overlap or intersect with any already activated virtual memory address.
