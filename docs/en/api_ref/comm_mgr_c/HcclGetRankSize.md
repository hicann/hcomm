# HcclGetRankSize

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-14T08:50:49.217Z pushedAt=2026-08-15T06:59:34.611Z -->

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

Queries the total number of ranks in a specified communicator.

## Function Prototype

```c
HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| rankSize | Output | Output address pointer to the total number of ranks in the specified collective communicator. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Initialize the communicator.
HcclComm comm;
// Obtain the number of ranks in the communicator.
uint32_t rankSize;
HcclGetRankSize(comm, &rankSize);
```
