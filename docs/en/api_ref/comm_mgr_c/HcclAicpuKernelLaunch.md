# HcclAicpuKernelLaunch

<!-- md-trans-meta sourceCommit=f059310bc8b19a0ab23cbea1d59204e99e0448cb translatedAt=2026-08-14T08:27:42.962Z pushedAt=2026-08-14T09:00:51.146Z -->

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

Runs a kernel function on the AI CPU. This API is used to deliver custom communication operators to the AI CPU for execution, and supports both group communication scenarios and non-group communication scenarios.

In group communication scenarios (where HcclGroupStart is called), this API adds the task to the group task queue and waits for HcclGroupEnd to execute it collectively. In non-group communication scenarios, the kernel function is directly run on the AI CPU.

## Function Prototype

```c
HcclResult HcclAicpuKernelLaunch(HcclComm comm, const HcclOpDesc *opInfo, const HcclKernelFuncInfo *funcInfo, ThreadHandle aicpuThreadHandle, aclrtStream userStream, const HcclKernelLaunchCfg *kernelLaunchCfg);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed. |
| opInfo | Input | Operator description parameters, including parameters related to the operator type and name.<br>For the definition of the HcclOpDesc type, see the data type [HcclOpDesc](./data_type_definition/HcclOpDesc.md). |
| funcInfo | Input | Kernel function information, including the dynamic library name, function name, parameters, and parameter size.<br>For the definition of the HcclKernelFuncInfo type, see the data type [HcclKernelFuncInfo](./data_type_definition/HcclKernelFuncInfo.md). |
| aicpuThreadHandle | Input | Handle of the AI CPU communication main thread.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../comm_opdev/datatype_definition/ThreadHandle.md) |
| userStream | Input | Stream used by the current rank. |
| kernelLaunchCfg | Input | Configuration for running the kernel function, including parameters such as the timeout period.<br>For the definition of the HcclKernelLaunchCfg type, see the data type [HcclKernelLaunchCfg](./data_type_definition/HcclKernelLaunchCfg.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- Used in custom communication operator scenarios (currently only the Send and Receive operators are supported).
- Ensure that the communicator has been correctly initialized before calling this API.

## Example

```c
// Initialize the operator description parameters.
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
aclrtStream userStream;
aclrtCreateStream(&userStream);
ThreadHandle aicpuThreadHandle;
HcclResult ret = HcclThreadAcquire(comm, engine, 1, 1, &aicpuThreadHandle);
if (ret != HCCL_SUCCESS) {
    // Handle the error.
}

HcclOpDesc opInfo;
HcclOpDescInit(&opInfo);
opInfo.opDescType = 1;  // Represent the Send or Receive operator.
opInfo.p2p.buffer = sendBuffer;
opInfo.p2p.cmdType = HCCL_CMD_SEND;
opInfo.p2p.dataType = HCCL_DATA_TYPE_FP32;
opInfo.p2p.count = dataSize;
opInfo.p2p.remoteRank = destRank;
opInfo.p2p.unfoldStream = userStream;

// Initialize the kernel function information.
HcclKernelFuncInfo funcInfo;
strncpy_s(funcInfo.kernelSoName, HCCL_KERNEL_SO_NAME_MAX_LEN, "libscatter_aicpu_kernel.so", strlen("libscatter_aicpu_kernel.so") + 1);
strncpy_s(funcInfo.kernelFuncName, HCCL_KERNEL_FUNC_NAME_MAX_LEN, "HcclLaunchP2pAicpuKernel", strlen("HcclLaunchP2pAicpuKernel") + 1);
funcInfo.args = kernelArgs;
funcInfo.argSize = argsSize;

// Initialize the kernel function configuration.
AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
u16 kernelLaunchTimeout = IsHcommDefaultTimeoutSupported() ? timeout.kernelLaunchTimeout :
ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
HcclKernelLaunchCfg kernelLaunchCfg;
HcclKernelLaunchCfgInit(&kernelLaunchCfg);
kernelLaunchCfg.timeOut = kernelLaunchTimeout;

// Run the AI CPU kernel function.
ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
if (ret != HCCL_SUCCESS) {
    // Handle the error.
}
```
