# HcclRankGraphGetRanksByLayer

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:40:38.477Z pushedAt=2026-08-17T08:25:31.040Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Given a communicator and a topology layer ID, returns the list of all rank IDs and the number of ranks in the topology instance where the current rank resides at that layer.

![Topology model](figures/topo_model.png)

Using the preceding topology model as an example:

- Layer 0 contains two topology instances. For ease of understanding, the topology instance IDs are defined as 0 and 1.
- Layer 1 contains one topology instance.

Assume that this API is called on rank 0. If layer 0 is specified, the API returns the rank ID list \[0,1,2\] and the rank quantity 3. If layer 1 is specified, the API returns the rank ID list \[0,1,2,3,4,5\] and the rank quantity 6.

## Function Prototype

```c
HcclResult HcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t **ranks, uint32_t *rankNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| ranks | Output | Rank ID list. |
| rankNum | Output | Rank quantity. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.

## Example

The topology model in [Description](#description) is used as an example.

For rank0:

```c
HcclComm commTp;
uint32_t* ranks = nullptr;
uint32_t rankNum;
HcclRankGraphGetRanksByLayer(commTp, netLayer=0, &ranks, &rankNum);
// For layer-0 topology, ranks=[0,1,2], rankNum=3
HcclRankGraphGetRanksByLayer(commTp, netLayer=1, &ranks, &rankNum);
// For layer-1 topology, ranks=[0,1,2,3,4,5], rankNum=6
```

For rank3:

```c
HcclComm commTp;
uint32_t* ranks = nullptr;
uint32_t rankNum;
HcclRankGraphGetRanksByLayer(commTp, netLayer=0, &ranks, &rankNum);
// For layer-0 topology, ranks=[3,4,5], rankNum=3
HcclRankGraphGetRanksByLayer(commTp, netLayer=1, &ranks, &rankNum);
// For layer-1 topology, ranks=[0,1,2,3,4,5], rankNum=6
```
