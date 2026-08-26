# HcommCcuKernelRegisterStart

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T09:20:28.174Z pushedAt=2026-08-17T06:09:32.486Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->

## Description

Marks the start of a kernel registration round and clears the set of pending kernels on the specified CCU instance, preparing for subsequent calls to [HcommCcuKernelRegister](HcommCcuKernelRegister.md).

A CCU instance supports multiple registration rounds. Each round starts with this API and ends with [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md). Within each round, [HcommCcuKernelRegister](HcommCcuKernelRegister.md) can be called to register one or more kernels.

## Function Prototype

//ccu_launch.h

```c
CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle);
```

## Parameters

| Name | Type | Description |
| --- | --- | --- |
| insHandle | Input | CCU instance handle, obtained from the HCCL communicator. Valid handles start from 1. Passing 0 or an unregistered handle returns `CCU_E_PTR`. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | `insHandle` is 0 or invalid, and no corresponding instance is found. |
| `CCU_E_INTERNAL` | Sequence error: This API is called again to start a new registration round before the previous registration round is ended by calling [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md). |

## Constraints

- CcuInsHandle must be obtained in the HCCL communicator first, and must be obtained before [HcommCcuKernelRegister](HcommCcuKernelRegister.md).
- This API and [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md) must be called in pairs: before starting a new registration round, the previous round must have been ended by calling [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md); otherwise, this API returns `CCU_E_INTERNAL`.
- This API can be called only on the host side, not inside a kernel function body.

## Example

```c
// insHandle is obtained from the HCCL communicator.
CcuInsHandle insHandle = 0;
// ... The call to HcommCcuInsCreate is omitted here ...

// Start a round of kernel registration.
CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelRegisterStart failed, ret = %d\n", ret);
    return ret;
}
// Subsequently call HcommCcuKernelRegister to register one or more Kernels.
```
