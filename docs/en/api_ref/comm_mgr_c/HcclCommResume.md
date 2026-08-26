# HcclCommResume

<!-- md-trans-meta sourceCommit=70ceaf3247c85b1c56f8a58d924d0411f8477047 translatedAt=2026-08-14T08:37:32.317Z pushedAt=2026-08-14T09:59:55.708Z -->

> [!NOTE] Note
> This API is reserved and may be changed in the future. It is not supported for developers.

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Restores the state of a communicator.

If a developer suspends a communicator by calling [HcclCommSuspend](HcclCommSuspend.md) or the aclrtDeviceTaskAbort API provided by ACL, after fault recovery, this API must be called to restore the communicator to the normal state.

## Function Prototype

```c
HcclResult HcclCommResume(HcclComm comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator to be restored from the suspended state to the normal state.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- Before calling this API, call the aclrtDeviceTaskAbort API provided by ACL to stop task execution on the current device.
- Before calling this API to restore the communicator state, perform a cluster synchronization operation.

## Example

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// Generate the rank identifier of the root node.
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));
// Initialize the communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm));
// Assume that the communicator has been suspended by the HcclCommSuspend API or the aclrtDeviceTaskAbort API provided by ACL. Resume the communicator.
HCCLCHECK(HcclCommResume(hcclComm));
// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
