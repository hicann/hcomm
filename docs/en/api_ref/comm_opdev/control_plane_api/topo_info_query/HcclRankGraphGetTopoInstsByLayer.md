# HcclRankGraphGetTopoInstsByLayer

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:41:39.482Z pushedAt=2026-08-17T08:35:55.717Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Given a communicator and a topology layer ID, queries the set of topology instances where the current rank resides.

## Function Prototype

```c
HcclResult HcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t **topoInsts, uint32_t *topoInstNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| topoInsts | Output | Topology instance list. |
| topoInstNum | Output | Number of topology instances. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.

## Example

```c
HcclComm comm;
uint32_t netLayer = 0;
uint32_t *topoInsts = nullptr;
uint32_t topoInstNum = 0;
// Default to 0 when topoInstanceId is not configured in the topo file.
HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum); 
// topoInsts = [0], topoInstNum = 1
```
