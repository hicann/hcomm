# HcclGetCommAsyncError

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T08:45:10.150Z pushedAt=2026-08-15T06:37:48.060Z -->

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
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

When the device NIC communication link in the cluster is unstable or network congestion occurs, "error cqe" is printed in the device log. This error is called an "RDMA ERROR CQE" error.

In the current version, this API can only be used to query whether an "RDMA ERROR CQE" error exists in the communicator.

> [!NOTE] Note
> This API is a synchronous API. That is, after the API is called, you need to wait for the result to be returned.

## Function Prototype

```c
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator to be queried for error information.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| asyncError | Output | - 0: No error occurs in the communicator.<br>  - 21: An "RDMA ERROR CQE" error occurs in the communicator. |

## Return Value

See the [HcclResult](./data_type_definition/HcclResult.md) type. In the current version, only the HCCL_E_REMOTE error type is returned.

## Constraints

- This API can be called only after a communicator is established.
- This API cannot be called after the communicator is destroyed.
