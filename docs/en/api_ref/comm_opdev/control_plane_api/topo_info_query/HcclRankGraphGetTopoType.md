# HcclRankGraphGetTopoType

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T09:43:39.300Z pushedAt=2026-08-17T08:37:34.267Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Given a communicator and a topology layer ID, queries the topology type corresponding to the current rank in the specified topology instance.

## Function Prototype

```c
HcclResult HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo *topoType)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| topoInstId | Input | Topology instance ID. |
| topoType | Output | Topology type.<br>For the definition of the CommTopo type, see [CommTopo](../../datatype_definition/CommTopo.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm;
CommTopo topoType;
HcclRankGraphGetTopoType(comm, netLayer=0, topoInstId=0, &topoType); // topoType=1 (1DMesh)
```
