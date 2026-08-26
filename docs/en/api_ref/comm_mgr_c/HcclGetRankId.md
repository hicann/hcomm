# HcclGetRankId

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-14T08:49:43.960Z pushedAt=2026-08-15T06:57:39.612Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
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

Obtains the rank ID of a device in a specified communicator.

## Function Prototype

```c
HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>For details about the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| rank | Output | Output address pointer of the rank ID in the specified collective communicator. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Initialize the communicator.
HcclComm comm;
// Obtain the rank ID of the current device.
uint32_t rank;
HcclGetRankId(comm, &rank);
```
