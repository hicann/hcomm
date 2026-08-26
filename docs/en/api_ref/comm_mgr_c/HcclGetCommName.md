# HcclGetCommName

<!-- md-trans-meta sourceCommit=bb004002a3fe9083a856ce984ccd88861a6e44cd translatedAt=2026-08-14T08:47:34.164Z pushedAt=2026-08-15T06:46:30.426Z -->

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

Obtains the name of the communicator where the specified communication operation is performed.

## Function Prototype

```c
HcclResult HcclGetCommName(HcclComm comm, char* commName)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| commName | Output | Name of the communicator obtained.<br>Type: char*. The maximum supported length is 128. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Initialize the communicator.
HcclComm comm;
// Query the communicator name.
char commName[128] = {0};
HcclResult ret = HcclGetCommName(comm, commName);
// Error handling
if (ret != HCCL_SUCCESS) {
}
```
