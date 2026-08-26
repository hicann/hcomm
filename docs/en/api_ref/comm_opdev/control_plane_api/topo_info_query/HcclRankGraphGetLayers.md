# HcclRankGraphGetLayers

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:38:29.274Z pushedAt=2026-08-17T08:09:23.223Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Queries the topology layer number list that contains the current rank and the number of topology layers.

![Topology model](figures/topo_model.png)

Taking the preceding topology model as an example, it contains two topology layers: Layer 0 and Layer 1. After this API is called, the returned topology layer number list is \[0,1\], and the number of topology layers is 2.

## Function Prototype

```c
HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the current rank resides.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayers | Output | Topology layer number list. |
| netLayerNum | Output | Number of topology layers. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The returned memory is managed by the library and must not be released by the caller.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.

## Example

Take the topology model in [Description](#description) as an example:

```c
HcclComm comm;
uint32_t *netLayers;
uint32_t layerNum;
HcclRankGraphGetLayers(comm, &netLayers, &layerNum);
// netLayers=[0,1], layerNum=2
```
