# HcclRankGraphGetRankSizeByLayer

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T09:41:37.741Z pushedAt=2026-08-17T08:31:53.085Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Given a communicator and a topology layer ID, returns the number of ranks in the topology instance where the current rank resides at that layer.

![Topology model](figures/topo_model.png)

Take the preceding topology model as an example:

- Layer 0 contains two topology instances. For ease of understanding, the topology instance IDs are defined as 0 and 1.
- Layer 1 contains one topology instance.

Assume that this API is called on rank 0. If layer 0 is specified, the API returns 3 as the number of ranks. If layer 1 is specified, the API returns 6 as the number of ranks.

## Function Prototype

```c
HcclResult HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| rankNum | Output | Number of ranks. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

Take the topology model in [Description](#description) as an example:

```c
HcclComm comm;
uint32_t rankNum;
HcclRankGraphGetRankSizeByLayer(comm, 0, &rankNum);
// rankNum=3
HcclRankGraphGetRankSizeByLayer(comm, 1, &rankNum);
// rankNum=6
```
