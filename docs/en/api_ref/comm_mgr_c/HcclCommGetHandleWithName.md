# HcclCommGetHandleWithName

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-11T07:48:28.460Z pushedAt=2026-08-11T10:41:45.869Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Not supported
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

Obtains a handle to the corresponding communicator based on the communicator name.

## Function Prototype

```c
HcclResult HcclCommGetHandleWithName(const char* commName, HcclComm* comm)
```

## Parameters

| Parameter Name | Input/Output | Description |
| --- | --- | --- |
| commName | Input | Communicator name.<br>const char* type, with a maximum length of 128. |
| comm | Output | Handle of the obtained communicator.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Define the communicator name for which the handle is to be obtained.
char commName[128] = "group_name_0";
HcclComm comm;
// Obtain the communicator handle corresponding to the communicator name.
HcclCommGetHandleWithName(commName, &comm);
```
