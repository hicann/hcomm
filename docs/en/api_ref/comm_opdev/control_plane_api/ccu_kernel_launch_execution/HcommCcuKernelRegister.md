# HcommCcuKernelRegister

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T09:19:16.952Z pushedAt=2026-08-17T06:03:12.134Z -->

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

Registers a CCU kernel function. This API executes the user-provided kernel function once on the host side. All `Ccu*` APIs in the function body do not perform actual hardware operations; they only record the complete operation sequence of the kernel. Upon successful registration, a kernel handle is returned for subsequent use by [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md).

If registration fails, the framework automatically rolls back the current kernel, and other registered kernels are not affected. If a `CCU_IF` branch in the kernel function body is not explicitly paired with `CCU_ELSE`, it is automatically closed when this `HcommCcuKernelRegister` call ends.

## Function Prototype

//ccu_launch.h

```c
CcuResult HcommCcuKernelRegister(CcuInsHandle insHandle, uint32_t dieId,
    const char *kernelFuncName, const void *kernelFunc,
    const void **kernelArgs, uint32_t argNum, CcuKernelHandle *kernelHandle);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| insHandle | Input | CCU instance handle, obtained from the HCCL communicator. Valid handles start from 1. Passing 0 or an unregistered handle returns `CCU_E_PTR`. |
| dieId | Input | Reserved parameter. This field is not used in the current implementation; the die actually used by the operator is automatically derived from the channel parameter information. In later versions, it will be used to specify the die on which the kernel runs. Currently, passing any value (for example, `0`) does not affect the behavior. |
| kernelFuncName | Input | Kernel name string, used for performance analysis and debugging identification. A null pointer or an empty string is allowed, in which case the default name `"CCU_KERNEL"` is used. If the length exceeds 128 characters, it is truncated to the first 128 characters. This string may differ from the symbol name of the kernel function. |
| kernelFunc | Input | Pointer to the kernel function. It cannot be a null pointer. |
| kernelArgs | Input | Pointer array of kernel function parameters. When `argNum` is not `0`, it cannot be a null pointer, and the first element `kernelArgs[0]` actually used cannot be null either. When `argNum` is `0`, this parameter is ignored and `nullptr` can be passed. During registration, `kernelArgs[0]` is passed to the kernel function body as is, and scalar values read through this parameter in the kernel function body are fixed as immediate values. Scalars that need to change at runtime must be passed through `taskArgs` and `CcuLoadArg`. |
| argNum | Input | Number of kernel function parameters. Currently only `0` or `1` is supported: `0` indicates a kernel with no parameters, and `1` indicates a kernel with a single parameter. Passing a value greater than `1` returns `CCU_E_PARA`. |
| kernelHandle | Output | Output parameter. The kernel handle is written back upon successful registration. It cannot be a null pointer. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | `insHandle` is 0 or invalid, `kernelFunc` is a null pointer, or `kernelArgs` (including `kernelArgs[0]`) is a null pointer when `argNum` is not `0`. |
| `CCU_E_PARA` | `argNum` is greater than `1` (currently only `0` or `1` is supported). |
| `CCU_E_UNAVAIL` | Hardware resources are insufficient, and the resources requested by this kernel exceed the available quota of the CCU instance. |
| `CCU_E_INTERNAL` | An internal error occurs, or the kernel function throws an uncaught exception; or a sequence error occurs: this API is called without first calling [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) to start a registration round. |

## Constraints

- This API must be called after [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) and before [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md); if [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) is not called first, this API returns `CCU_E_INTERNAL`.
- This API can be called multiple times within the same registration round of the same instance, with each call registering an independent kernel.
- `argNum` currently supports only `0` or `1`; when it is `1`, only `kernelArgs[0]` takes effect and the remaining elements are ignored.
- The scalar pointed to by `kernelArgs[0]` is read during the registration phase and fixed as an immediate value. If the value of a scalar needs to be specified dynamically at each launch, it must be passed through the `taskArgs` array and the `CcuLoadArg` API inside the kernel, rather than through `kernelArgs`.
- `dieId` is a reserved parameter and is not used in the current implementation.
- This API can only be called on the host side and cannot be called in a nested manner (that is, it cannot be called again inside the kernel function body).

## Example

```c
// User-defined kernel input parameter structure
typedef struct {
    uint32_t loopCount;
} MyKernelArg;

// User-defined kernel function that calls the Ccu* series APIs in its function body
CcuResult MyKernel(CcuKernelArg arg)
{
    // CcuKernelArg is void* (see ccu_types.h). You can define a custom input parameter struct and pass it through kernelArgs.
    MyKernelArg *myArg = (MyKernelArg *)arg;
    // Call the CCU data plane programming APIs here, such as LoadArg.
    // ...
    return CCU_SUCCESS;
}

// Obtain insHandle from the communicator.
CcuInsHandle insHandle = 0;
uint32_t dieId = 0;                     // Reserved parameter, not used in the current implementation.
MyKernelArg arg = { .loopCount = 10 };
const void *kernelArgs[] = { &arg };    // Input parameter pointer array.
uint32_t argNum = 1;                    // Currently only 0 or 1 is supported.
CcuKernelHandle kernelHandle = 0;

CcuResult ret = HcommCcuKernelRegister(
    insHandle,
    dieId,                          // Reserved die ID. Pass 0.
    "MyKernel",                     // Kernel name, used for debugging and can be empty.
    (const void *)MyKernel,         // Kernel function pointer.
    kernelArgs,                     // Kernel input parameter pointer array.
    argNum,                         // Number of input parameters
    &kernelHandle);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelRegister failed, ret = %d\n", ret);
    return ret;
}
```
