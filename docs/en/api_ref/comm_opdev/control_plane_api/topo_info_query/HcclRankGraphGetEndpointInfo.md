# HcclRankGraphGetEndpointInfo

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:38:05.147Z pushedAt=2026-08-17T08:01:38.044Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Obtains the topology attribute information of a specified communication device.

## Function Prototype

```c
HcclResult HcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc *endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void *info)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| rankId | Input | Rank ID to which the endpoint to be queried belongs. |
| endpointDesc | Input | Endpoint descriptor, which is the "endpointDesc" obtained through the [HcclRankGraphGetEndpointDesc](HcclRankGraphGetEndpointDesc.md) API. |
| endpointAttr | Input | Type of the endpoint attribute to be queried.<br>For the definition of the EndpointAttr type, see [EndpointAttr](../../datatype_definition/EndpointAttr.md). |
| infoLen | Input | Size of the provided info buffer (in bytes). |
| info | Output | Pointer to the output buffer that stores the attribute information. |

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

// Obtain the topology attribute information.
uint32_t rankId = 0; // Rank ID of the endpoint to be queried
EndpointAttrBwCoeff bwCoeff = {0};
uint32_t size = sizeof(EndpointAttrBwCoeff); // Must be equal to the target type size
HcclRankGraphGetEndpointInfo(comm, rankId, endpointDesc, ENDPOINT_ATTR_BW_COEFF, size, &bwCoeff);
```
