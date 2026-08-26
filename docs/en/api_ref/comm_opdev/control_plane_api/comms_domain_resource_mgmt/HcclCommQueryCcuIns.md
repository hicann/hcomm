# HcclCommQueryCcuIns

<!-- md-trans-meta sourceCommit=97c142fbd6f7bfa37c6fcae34433680b079af61d translatedAt=2026-08-14T09:27:02.678Z pushedAt=2026-08-17T06:53:39.926Z -->

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

Queries the CCU instance bound to a specified HCCL communicator and returns the instance handle for subsequent registration and dispatch of CCU kernels.

Currently, a communicator can bind at most one CCU instance. Therefore, when the query succeeds, exactly one instance is returned: insHandles[0] is the bound instance handle, and *insNum is 1.

> [!NOTE] Note
> The query result is for borrowing only and does not transfer ownership of the instance. The returned CCU instance is owned by the communicator, which is responsible for releasing it. The caller must not destroy the instance; otherwise, the same instance will be released repeatedly, corrupting the resource management of the communicator.

## Function Prototype

```c
HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle, which must not be nullptr.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| insHandles | Output | Array of CCU instance handles, which must not be nullptr. After a successful query, insHandles[0] returns the bound CCU instance handle.<br>The CcuInsHandle type is defined as follows:<br>typedef uint64_t CcuInsHandle; |
| insNum | Output | Number of CCU instances, which must not be nullptr. After a successful query, the returned value is fixed at 1. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

- When comm, insHandles, or insNum is nullptr, HCCL_E_PTR is returned.
- When the communicator generation does not support CCU (earlier than Ascend 950PR/Ascend 950DT), HCCL_E_NOT_SUPPORT is returned.
- When the communicator has no CCU instance bound, HCCL_E_UNAVAIL is returned.

## Constraints

1. The CCU feature is supported only on Ascend 950PR/Ascend 950DT and later generations. Calling this API on a communicator of an earlier generation returns HCCL_E_NOT_SUPPORT.
2. This API must be called after the communicator is initialized and bound to a CCU instance. Otherwise, HCCL_E_UNAVAIL is returned.
3. The returned CCU instance handle is borrowed only. Its ownership remains with the communicator, which is responsible for releasing it. The caller must not destroy the instance. Otherwise, the same instance will be released repeatedly, corrupting the resource management of the communicator.

## Example

```c
// comm must be initialized through APIs such as HcclCommInitClusterInfo. This is only a placeholder for the example.
HcclComm comm = /*Initialized communication domain handle*/;
CcuInsHandle insHandle = 0;
uint32_t insNum = 0;

// Query the CCU instance bound to the communicator.
HcclResult ret = HcclCommQueryCcuIns(comm, &insHandle, &insNum);
// The current implementation always returns one CCU instance, and insNum != 1 is treated as a failure.
if (ret != HCCL_SUCCESS || insNum != 1) {
    // Error handling: if the API fails, return the original error code; if the API succeeds but the instance quantity is abnormal, return the internal error code.
    return (ret != HCCL_SUCCESS) ? ret : HCCL_E_INTERNAL;
}

// Use insHandle to register and dispatch the CCU kernel.
CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
// ...
```
