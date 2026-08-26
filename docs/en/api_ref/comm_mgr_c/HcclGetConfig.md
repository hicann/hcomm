# HcclGetConfig

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T08:48:23.347Z pushedAt=2026-08-15T06:49:59.630Z -->

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

Obtains the configuration related to collective communication.

## Function Prototype

```c
HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue *configValue)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Input | Collective communication configuration parameter.<br>Type [HcclConfig](./data_type_definition/HcclConfig.md). In the current version, only "HCCL_DETERMINISTIC" is supported. |
| configValue | Output | Value of the parameter configured in config.<br>For details, see [HcclConfigValue](./data_type_definition/HcclConfigValue.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Query the deterministic computation switch.
HcclConfig config = HCCL_DETERMINISTIC;
union HcclConfigValue configValue;
HcclGetConfig(HCCL_DETERMINISTIC, &configValue);
```
