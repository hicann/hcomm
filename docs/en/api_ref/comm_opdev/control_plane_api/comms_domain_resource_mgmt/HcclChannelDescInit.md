# HcclChannelDescInit

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:23:39.140Z pushedAt=2026-08-17T06:30:10.935Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Initializes a communication channel descriptor list.

## Function Prototype

```c
HcclResult HcclChannelDescInit(HcclChannelDesc *channelDesc, uint32_t descNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channelDesc | Output | Communication channel descriptor list. The list length is descNum. This API initializes the structure.<br>For details about the HcclChannelDesc type, see [HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md). |
| descNum | Input | Number of communication channel descriptors. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

The HcclChannelDesc structure must be initialized by calling this API.

## Example

The following example initializes a communication channel descriptor list with two communication channel descriptors:

```c
uint32_t channelNum = 2;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);
```
