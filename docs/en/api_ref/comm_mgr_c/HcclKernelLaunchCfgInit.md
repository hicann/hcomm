# HcclKernelLaunchCfgInit

<!-- md-trans-meta sourceCommit=f059310bc8b19a0ab23cbea1d59204e99e0448cb translatedAt=2026-08-14T08:53:57.722Z pushedAt=2026-08-15T07:30:15.862Z -->

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
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Initializes the HcclKernelLaunchCfg structure to describe the configuration information for running a kernel function on the AI CPU.

This API first fills the entire structure with 0xFF, and then sets the ABI header information (including the version number, magic number, and structure size) to put the structure into a usable initial state. After calling this API, developers need to set fields such as timeOut as required by the actual task, and then pass the configuration to [HcclAicpuKernelLaunch](./HcclAicpuKernelLaunch.md) to dispatch the task.

## Function Prototype

```c
static inline HcclResult HcclKernelLaunchCfgInit(HcclKernelLaunchCfg *kernelLaunchCfg)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| kernelLaunchCfg | Output | KernelLaunch configuration parameters to be initialized.<br>For the definition of the HcclKernelLaunchCfg type, see the data type [HcclKernelLaunchCfg](./data_type_definition/HcclKernelLaunchCfg.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and HCCL_E_PTR when a null pointer is passed in.

## Constraints

- This API only initializes the ABI header information. The remaining fields of the structure are filled with 0xFF. After initialization, the caller needs to set the service fields (such as timeOut) as required.
- The pointer passed in must point to an HcclKernelLaunchCfg object with allocated memory and cannot be a null pointer.

## Example

```c
// Initialize the kernel function configuration.
HcclKernelLaunchCfg kernelLaunchCfg;
HcclKernelLaunchCfgInit(&kernelLaunchCfg);
kernelLaunchCfg.timeOut = 60;  // Set the timeout period, in seconds.

// Run the AI CPU kernel function.
HcclResult ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
if (ret != HCCL_SUCCESS) {
    // Handle the error.
}
```
