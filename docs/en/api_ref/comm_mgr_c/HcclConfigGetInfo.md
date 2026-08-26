# HcclConfigGetInfo

<!-- md-trans-meta sourceCommit=f78ffaa53d8d460b15f67dab0e39f1184082250b translatedAt=2026-08-14T08:44:20.108Z pushedAt=2026-08-15T05:59:55.438Z -->

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

Obtains the HCCL configuration information of a specified communicator.

Queries the configuration information based on the configuration item type and writes it into the buffer provided by the caller. Currently, only the expansion mode of communication operators can be queried.

## Function Prototype

```c
HcclResult HcclConfigGetInfo(HcclComm comm, HcclConfigType cfgType, uint32_t infoLen, void *info);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle. |
| cfgType | Input | Type of the configuration item to query. For the definition of HcclConfigType, see [HcclConfigType](./data_type_definition/HcclConfigType.md). |
| infoLen | Input | Size (in bytes) of the target configuration type, which must be equal to the actual size of the configuration type to query. |
| info | Output | Output buffer for the configuration information, which must be aligned to the target configuration type and writable. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
HcclConfigTypeOpExpansionMode mode;
uint32_t size = sizeof(HcclConfigTypeOpExpansionMode); // Must be equal to the size of the target type.
HcclResult ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, size, &mode);
```
