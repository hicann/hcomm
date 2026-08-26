# HcclRankGraphGetTopoTypeByLayer

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:43:57.086Z pushedAt=2026-08-17T08:40:29.760Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Given a communicator and a topology layer ID, returns the topology type of the layer where the current rank resides.

![Topology model](figures/topo_model.png)

Take the preceding topology model as an example:

- Layer 0 contains two topology instances. For ease of understanding, the topology instance IDs are defined as 0 and 1. The topology type of instance 0 is 1DMesh, and that of instance 1 is Clos.
- Layer 1 contains one topology instance, whose topology type is Clos.

## Function Prototype

```c
HcclResult HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo *topoType)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| topoType | Output | Topology type, including 1DMesh, Clos, and custom types.<br>For the definition of the CommTopo type, see [CommTopo](../../datatype_definition/CommTopo.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

Take the topology model in [Description](#description) as an example.

For rank 0:

```c
HcclComm comm;
CommTopo topoType;
HcclRankGraphGetTopoTypeByLayer(comm, 0, &topoType);  
// Layer0 topoType=1 (1DMesh)
HcclRankGraphGetTopoTypeByLayer(comm, 1, &topoType);  
// Layer1 topoType=0 (Clos)
```

For rank 3:

```c
HcclComm comm;
CommTopo topoType;
HcclRankGraphGetTopoTypeByLayer(comm, 0, &topoType);  
// Layer0 topoType=0 (Clos)
HcclRankGraphGetTopoTypeByLayer(comm, 1, &topoType);  
// Layer1 topoType=0 (Clos)
```
