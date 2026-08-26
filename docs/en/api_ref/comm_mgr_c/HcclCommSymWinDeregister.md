# HcclCommSymWinDeregister

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T08:38:34.705Z pushedAt=2026-08-15T01:29:53.344Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
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

Deregisters a registered symmetric memory window and releases the symmetric memory window resources. This API does not release the memory allocated by the user. The user still needs to release the corresponding memory in the same way as it was allocated.

For Atlas A3 training products/Atlas A3 inference products, this API supports the HCCS link communication scenario. For Ascend 950PR/Ascend 950DT, this API supports the URMA scenario.

## Function Prototype

```c
HcclResult HcclCommSymWinDeregister(HcclCommSymWindow winHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| winHandle | Input | Handle of the registered symmetric window.<br>For the definition of the HcclCommSymWindow type, see [HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- This API must be used together with [HcclCommSymWinRegister](HcclCommSymWinRegister.md).
- The supported scope is the same as that of [HcclCommSymWinRegister](HcclCommSymWinRegister.md): For Atlas A3 training products/Atlas A3 inference products, only the HCCS link communication scenario is supported; for Ascend 950PR/Ascend 950DT, only the URMA scenario is supported.
- Ensure that all ranks in the communicator call this API at the same time to release the symmetric window resources.

## Example

See [Example](HcclCommSymWinRegister.md#example).
