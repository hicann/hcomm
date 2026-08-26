# HcclSetConfig

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T08:56:09.217Z pushedAt=2026-08-15T07:40:48.270Z -->

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

Performs collective communication-related configuration. Currently, only the configuration of whether deterministic computation is supported is available.

When deterministic computation is not enabled, the results of multiple executions may differ. This difference generally arises because asynchronous multi-thread execution exists in the operator implementation, which causes the order of floating-point accumulation to change. When deterministic computation is enabled, the operator produces the same output across multiple executions under the same hardware and input.

By default, deterministic computation or the order-preserving function does not need to be enabled. However, when the results of multiple model executions differ or during precision tuning, you can enable deterministic computation or the order-preserving function to assist in debugging and tuning. Note that after these functions are enabled, the operator execution time increases, leading to performance degradation.

## Function Prototype

```c
HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Input | Parameters that can be configured in config.<br>Type [HcclConfig](./data_type_definition/HcclConfig.md). In the current version, only "HCCL_DETERMINISTIC" can be configured. |
| configValue | Input | Value of the parameter configured in config.<br>See the [HcclConfigValue](./data_type_definition/HcclConfigValue.md) type. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Enable deterministic computation.
HcclConfig config = HCCL_DETERMINISTIC;
union HcclConfigValue configValue;
configValue.value = 1;
HcclSetConfig(config, configValue);
```
