# HcclCommSuspend

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T08:38:27.286Z pushedAt=2026-08-14T10:08:17.959Z -->

> [!NOTE] Note
> This API is reserved and may change in the future. It is not intended for developer use.

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Not supported
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

When an on-chip memory uncorrectable memory error (UCE) occurs (the ACL API returns the ACL_ERROR_RT_DEVICE_MTE_ERROR error code), you can call this API to set the communicator to the suspended state.

This API suspends the communicator without exiting the host-side process. After the fault is rectified, you can call [HcclCommResume](HcclCommResume.md) to restore the communicator state.

## Function Prototype

```c
HcclResult HcclCommSuspend(HcclComm comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator to be suspended.<br>For details about the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- This API must be used together with [HcclCommResume](HcclCommResume.md).
- This API cannot be executed concurrently with APIs related to collective communication or point-to-point communication.
