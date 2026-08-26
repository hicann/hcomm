# HcclRankGraphGetEndpointDesc

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:36:30.073Z pushedAt=2026-08-17T07:58:27.255Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Obtains the endpoint description list of a topology instance.

## Function Prototype

```c
HcclResult HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t *descNum, EndpointDesc *endpointDesc);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| layer | Input | Topology layer number. |
| topoInstId | Input | Topology instance ID. |
| descNum | Input/Output | As an output, it indicates the number of communication device descriptions actually obtained.<br>As an input, it must be equal to the value of "num" output by [HcclRankGraphGetEndpointNum](HcclRankGraphGetEndpointNum.md). |
| endpointDesc | Output | Endpoint description list. The caller needs to allocate memory for it.<br>For the definition of the EndpointDesc type, see [EndpointDesc](../../datatype_definition/EndpointDesc.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Communicator handle
HcclComm comm;

// Obtain the number of endpoints.
uint32_t layer = 0;
uint32_t topoInstId = 0;
uint32_t num = 0;
HcclRankGraphGetEndpointNum(comm, layer, topoInstId, &num);

// Obtain the endpoint description list.
uint32_t descNum = num;
EndpointDesc endpointDesc[descNum];
HcclRankGraphGetEndpointDesc(comm, layer, topoInstId, &descNum, endpointDesc);
```
