# HcclRankGraphGetEndpointNum

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T09:38:09.218Z pushedAt=2026-08-17T08:03:06.373Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Obtains the number of endpoints in a topology instance.

## Function Prototype

```c
HcclResult HcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t *num)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| layer | Input | Topology layer ID. |
| topoInstId | Input | Topology instance ID. |
| num | Output | Number of endpoints returned. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm;
uint32_t layer = 0;
uint32_t topoInstId = 0;
uint32_t num = 0;
HcclRankGraphGetEndpointNum(comm, layer, topoInstId, &num);
```
