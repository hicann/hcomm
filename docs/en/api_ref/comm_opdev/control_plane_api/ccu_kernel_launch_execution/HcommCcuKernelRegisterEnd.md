# HcommCcuKernelRegisterEnd

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T09:19:36.447Z pushedAt=2026-08-17T06:00:54.776Z -->

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

Ends a round of kernel registration, translates all kernels generated during this registration round into CCU device instructions at once, and delivers them to the device memory. After the translation is complete, all kernels registered in this round can be launched for execution through [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md).

## Function Prototype

//ccu_launch.h

```c
CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| insHandle | Input | CCU instance handle, obtained from the HCCL communicator. Valid handles start from 1; passing 0 or an unregistered handle returns `CCU_E_PTR`. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | `insHandle` is 0 or invalid, and the corresponding instance is not found. |
| `CCU_E_INTERNAL` | Internal error: translation failed or device memory copy failed; or a sequence error: this API is called without first calling [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md). |
| `CCU_E_UNAVAIL` | Hardware resources are insufficient to complete resource allocation and translation for this round of kernel registration. |

## Constraints

- This API should be called after [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) and at least one [HcommCcuKernelRegister](HcommCcuKernelRegister.md) call, and before [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md). If [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) is not called first, this API returns `CCU_E_INTERNAL`.

> [!NOTE] Note
> The APIs must be called in the following order: [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) → [HcommCcuKernelRegister](HcommCcuKernelRegister.md) → this API. Calling this API without first calling [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) returns `CCU_E_INTERNAL`.

- After this API is successfully called, the kernels registered in this round can be launched independently. To register a new round of kernels, call [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) again to start a new round.
- This API can only be called on the host side, and cannot be called inside a kernel function body.

## Example

```c
// insHandle is obtained from the communicator, and RegisterStart and Register have been completed.
CcuInsHandle insHandle = 0;
// ... The calls to HcommCcuKernelRegisterStart and HcommCcuKernelRegister are omitted here ...

// End registration, translate to device instructions, and deliver them.
CcuResult ret = HcommCcuKernelRegisterEnd(insHandle);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelRegisterEnd failed, ret = %d\n", ret);
    return ret;
}
// All kernels registered in this round are ready. You can call HcommCcuKernelLaunch to start them.
```
