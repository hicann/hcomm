# HcclGetErrorString

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T08:48:58.849Z pushedAt=2026-08-15T06:54:03.083Z -->

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

Parses an error code of the HcclResult type.

## Function Prototype

```c
const char *HcclGetErrorString(HcclResult code)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| code | Input | Error code to be parsed, of the [HcclResult](./data_type_definition/HcclResult.md) type. |

## Return Value

Parsing result of the error code of the [HcclResult](./data_type_definition/HcclResult.md) type.

## Constraints

None

## Example

```c
// Initialize device resources.
aclInit(NULL);
uint32_t devId = 0;
aclrtSetDevice(devId);

// Create a communicator.
HcclComm hcclComm;
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
HcclCommInitRootInfo(8, &rootInfo, 0, &hcclComm);

// Query asynchronous errors of the communicator and parse the error code.
HcclResult asyncError = HCCL_SUCCESS;
HcclGetCommAsyncError(hcclComm, &asyncError);
if (asyncError != HCCL_SUCCESS) {
    const char *errStr = HcclGetErrorString(asyncError);
    printf("comm async error: %s\n", errStr);
}

// Destroy the communicator.
HcclCommDestroy(hcclComm);
aclFinalize();
```
