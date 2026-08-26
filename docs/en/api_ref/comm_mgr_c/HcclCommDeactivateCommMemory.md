# HcclCommDeactivateCommMemory

<!-- md-trans-meta sourceCommit=ffc5b1340e1ba3599f61afdda8ea78f73c8e36d1 translatedAt=2026-08-14T08:31:08.180Z pushedAt=2026-08-14T09:15:48.623Z -->

> [!NOTE]Note
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

Deactivates the activated virtual memory. After deactivation, if the address is used for collective communication again, the zero-copy feature cannot be enabled.

## Function Prototype

```c
HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void *virPtr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. You are advised to use the largest communicator within the server, that is, the communicator covering the maximum number of devices. |
| virPtr | Input | Start address of the virtual address to be deactivated, that is, the virtual memory address specified by the "virPtr" parameter of the [HcclCommActivateCommMemory](HcclCommActivateCommMemory.md) API.<br>Note that the specified virtual memory must have been successfully activated, and only the entire address block can be deactivated. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None
