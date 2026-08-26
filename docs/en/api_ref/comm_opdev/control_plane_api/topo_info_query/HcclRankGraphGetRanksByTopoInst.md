# HcclRankGraphGetRanksByTopoInst

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:40:52.520Z pushedAt=2026-08-17T08:33:39.029Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Given a communicator and a topology layer ID, queries the rank information contained in the specified topology instance corresponding to the current rank.

## Function Prototype

```c
HcclResult HcclRankGraphGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t **ranks, uint32_t *rankNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| topoInstId | Input | Topology instance ID (existing in the topology file). |
| ranks | Output | List of ranks contained in the corresponding topology instance. |
| rankNum | Output | Number of ranks in the list. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.

## Example

```c
 // 8-device communicator, the same 8p Mesh
HcclComm comm;
uint32_t netlayer = 0;
uint32_t topoInstId = 0;
uint32_t *ranks;
uint32_t rankNum;
HcclRankGraphGetRanksByTopoInst( comm, netLayer, topoInstId,  &ranks, &rankNum )
 // ranks = [0,1,2,…,7],  rankNum=8
```
