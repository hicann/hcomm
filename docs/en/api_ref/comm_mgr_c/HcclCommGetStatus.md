# HcclCommGetStatus

<!-- md-trans-meta sourceCommit=5f3dd200278c2c644e98d7839407bd8e2e1b26ec translatedAt=2026-08-14T08:31:20.626Z pushedAt=2026-08-14T09:24:38.577Z -->

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

Obtains the communicator status during operator dispatch to determine whether an operator can be dispatched.

## Function Prototype

```c
HcclResult HcclCommGetStatus(const char *commId, HcclCommStatus *status)
```

## Parameters

| Name | Input/Output | Description |
| --- | --- | --- |
| commId | Input | Communicator name.<br>const char* type, with a maximum length of 128. |
| status | Output | Communicator status. For the definition of the HcclCommStatus type, see [HcclCommStatus](./data_type_definition/HcclCommStatus.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

Used in custom communication operator scenarios.

## Example

```c
HcclComm comm;
char commName[128];
HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;

... //Create a communicator.

HCCLCHECK(HcclGetCommName(comm, commName));
HCCLCHECK(HcclCommGetStatus(commName, &commStatus));
```
