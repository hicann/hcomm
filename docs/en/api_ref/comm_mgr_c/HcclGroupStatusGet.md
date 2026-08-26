# HcclGroupStatusGet

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T08:53:06.615Z pushedAt=2026-08-15T07:31:12.525Z -->

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

Obtains the group feature status and determines whether this feature is enabled.

## Function Prototype

```c
HcclResult HcclGroupStatusGet(bool *isGroupEnabled)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| isGroupEnabled | Output | Group status. TRUE indicates that the feature is enabled, and FALSE indicates that the feature is disabled. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
bool isGroupEnabled = false;
HCCLCHECK(HcclGroupStatusGet(&isGroupEnabled));
```
