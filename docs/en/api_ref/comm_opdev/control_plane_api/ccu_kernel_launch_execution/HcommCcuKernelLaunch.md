# HcommCcuKernelLaunch

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T09:18:11.376Z pushedAt=2026-08-17T03:43:51.742Z -->

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

Starts the execution of a CCU Kernel. This API submits a translated kernel to the CCU hardware for instruction fetch and execution. The same kernel handle can be used to call this API multiple times, with different `taskArgs` for each call, implementing "register once, launch multiple times".

## Function Prototype

//ccu_launch.h

```c
CcuResult HcommCcuKernelLaunch(ThreadHandle threadHandle,
    CcuKernelHandle kernelHandle, const void *taskArgs, uint32_t argNum);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| threadHandle | Input | Communication thread handle, which must be a valid handle obtained through [HcclThreadAcquireWithStream](../comms_domain_resource_mgmt/HcclThreadAcquireWithStream.md). The value cannot be 0. |
| kernelHandle | Input | Kernel handle, which must be a valid handle obtained through [HcommCcuKernelRegister](HcommCcuKernelRegister.md). The value cannot be 0. |
| taskArgs | Input | Pointer to a `uint64_t` array. The array elements are indexed by the `argId` in `CcuLoadArg` and provide runtime variable parameters for the kernel. If `CcuLoadArg` is not used in the kernel, a null pointer can be passed. |
| argNum | Input | Number of elements in the `taskArgs` array, in units of `uint64_t` elements instead of bytes. The value must equal the total number of distinct `argId` values used during the kernel registration phase. If `CcuLoadArg` is not used in the Kernel, the value is 0. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PARA` | Parameter error. `threadHandle` or `kernelHandle` is 0. |
| `CCU_E_PTR` | This error return value includes the following cases: the kernel corresponding to `kernelHandle` does not exist or the handle is invalid, the thread stream is empty, or `taskArgs` is a null pointer when `argNum > 0`. |
| `CCU_E_INTERNAL` | Internal error. `argNum` is inconsistent with the total number of different `argId` values used by `CcuLoadArg` in the kernel, or internal dispatch fails. |

## Constraints

- This API must be called after [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md).
- The unit of `argNum` is the number of `uint64_t` elements, not the number of bytes. Passing the number of bytes causes the `CCU_E_INTERNAL` error.
- When there are many `taskArgs` parameters, the framework automatically dispatches them in batches without user intervention.
- This API can only be called on the host side, not inside the Kernel function body.

## Example

```c
// threadHandle is returned by HcclThreadAcquireWithStream.
// kernelHandle is returned by HcommCcuKernelRegister, and RegisterEnd has been completed.
ThreadHandle threadHandle = 0;
CcuKernelHandle kernelHandle = 0;
// ... The preceding calls are omitted here ...

// Start the kernel for the first time and pass runtime parameters (the number of elements, not the number of bytes).
uint64_t taskArgs[2] = { 100, 200 };
CcuResult ret = HcommCcuKernelLaunch(threadHandle, kernelHandle, taskArgs, 2);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelLaunch failed, ret = %d\n", ret);
    return ret;
}

// The same kernel can be started repeatedly, and different taskArgs can be passed each time.
taskArgs[0] = 300;
taskArgs[1] = 400;
ret = HcommCcuKernelLaunch(threadHandle, kernelHandle, taskArgs, 2);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelLaunch failed, ret = %d\n", ret);
    return ret;
}
```
