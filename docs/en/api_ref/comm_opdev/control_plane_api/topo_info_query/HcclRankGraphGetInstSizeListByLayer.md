# HcclRankGraphGetInstSizeListByLayer

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:38:17.018Z pushedAt=2026-08-17T08:06:20.736Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Given a communicator and a topology layer ID, queries the number of topology instances at the layer and the number of ranks contained in each instance.

![Topology model](figures/topo_model.png)

Take the preceding topology model as an example:

- Layer 0 contains two topology instances. For ease of understanding, the topology instance IDs are defined as 0 and 1, and each instance contains three ranks.
- Layer 1 contains one topology instance, which contains six ranks.

## Function Prototype

```c
HcclResult HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t **instSizeList, uint32_t *listSize)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| instSizeList | Output | List of the number of ranks contained in each topology instance at this layer. |
| listSize | Output | Size of instSizeList, that is, the number of topology instances contained at this layer. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.

## Example

Take the topology model in [Description](#description) as an example:

```c
HcclComm comm;
uint32_t *instSizeList;
uint32_t listSize;
HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize);
// instSizeList=[3,3], listSize=2
HcclRankGraphGetInstSizeListByLayer(comm, 1, &instSizeList, &listSize);
// instSizeList=[6], listSize=1
```
