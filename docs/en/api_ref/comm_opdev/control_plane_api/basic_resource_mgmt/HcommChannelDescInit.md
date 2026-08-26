# HcommChannelDescInit

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:09:01.556Z pushedAt=2026-08-17T02:10:33.384Z -->

## Supported Products

- Atlas 350 accelerator card: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Initializes the communication channel description list.

## Function Prototype

```c
HcommResult HcommChannelDescInit(HcommChannelDesc *channelDesc, uint32_t descNum)
```

## Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| channelDesc | Input/Output | Communication channel description list. The list length is descNum. The function initializes this structure.<br>For details about the HcommChannelDesc type, see [HcommChannelDesc](../../datatype_definition/HcommChannelDesc.md). |
| descNum | Input | Number of communication channel descriptions. |

## Return Value

HcommResult: This API returns 0 on success and a non-zero value on failure.

## Constraints

- The HcommChannelDesc structure must be initialized by calling this API.

## Example

The following example initializes a communication channel description list that contains two communication channel descriptions:

```c
uint32_t channelNum = 2;
std::vector<HcommChannelDesc> channelDesc(channelNum);
HcommChannelDescInit(channelDesc.data(), channelNum);
```
