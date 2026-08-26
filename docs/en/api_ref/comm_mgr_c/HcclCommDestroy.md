# HcclCommDestroy

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-11T07:47:52.815Z pushedAt=2026-08-11T10:27:57.315Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 Training Series/Atlas A3 Inference Series: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 Training Series/Atlas A2 Inference Series: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas Inference Series: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas Training Series: Supported
<!-- end id5 -->

## Description

Destroys a specified HCCL communicator.

## Function Prototype

```c
HcclResult HcclCommDestroy(HcclComm comm)
```

## Parameters

| Parameter Name | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Pointer to the communicator to be destroyed.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- This API supports cross-thread call:
  - When the communicator is in a link establishment stuck state or unoccupied state, cross-thread call of this API to destroy the communicator is supported, and HCCL_SUCCESS is returned.

    After the communicator is successfully destroyed, the ongoing communication operators will directly exit with an error without waiting for the timeout period, and an ERROR-level log is printed. The log keyword is "Terminating operation due to external request".

  - When the communicator is not in a link establishment stuck state, or is in another occupied state (for example, during communicator link establishment or communication operator execution), cross-thread call of this API returns the HCCL_E_AGAIN error, and a WARNING-level log is printed. The log keyword is "[HcclCommDestroy] comm is in use, please try again later".

- In multi-threaded scenarios, ensure the call sequence of HCCL APIs. After the communicator is destroyed by calling this API, other collective communication APIs are no longer supported.

## Example

```c
uint32_t rankSize = 2;
int32_t devices[rankSize] = {0, 1};
HcclComm comms[rankSize];
// Initialize the communicator.
HcclCommInitAll(rankSize, devices, comms);
// Destroy the communicator.
for (uint32_t i = 0; i &lt; rankSize; i++) {
    HcclCommDestroy(comms[i]);
}
```
