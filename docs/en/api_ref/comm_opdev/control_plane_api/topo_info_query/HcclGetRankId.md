# HcclGetRankId

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T09:35:44.895Z pushedAt=2026-08-17T07:53:48.116Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the rank ID of a device in a specified communicator.

## Function Prototype

```c
HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| rank | Output | Output address pointer of the rank ID in the communicator. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
uint32_t rank;
HcclComm comm;
HcclGetRankId(comm, &rank);
```
