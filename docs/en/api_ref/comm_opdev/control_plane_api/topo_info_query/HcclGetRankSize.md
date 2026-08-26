# HcclGetRankSize

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T09:35:47.459Z pushedAt=2026-08-17T07:54:59.548Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Queries the number of ranks in the specified communicator.

## Function Prototype

```c
HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| rankSize | Output | Number of ranks in the communicator. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

Take a communicator with 4 servers and 8 devices as an example. The total number of ranks is 32:

```c
uint32_t rankSize;
HcclComm comm;
HcclGetRankSize(comm, &rankSize);
// rankSize=32
```
