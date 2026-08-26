# HcclRankGraphGetLinks

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:39:11.800Z pushedAt=2026-08-17T08:14:06.502Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Queries the communication connection information between a source rank and a destination rank based on a given communicator and topology layer ID.

Take Atlas A3 training products/Atlas A3 inference products as an example:

- Example 1: The source rank and destination rank are in two different SuperPoDs.

  netLayer = 0, no connection.

  netLayer = 1, no connection.

  netLayer = 2, RDMA connection.

- Example 2: The source rank and destination rank are in the same SuperPoD but not in the same AI server.

  netLayer = 0, no connection.

  netLayer = 1, HCCS connection.

  netLayer = 2, no connection.

- Example 3: The source rank and destination rank are in the same AI server but not in the same NPU.

  netLayer = 0, HCCS connection.

  netLayer = 1, no connection.

  netLayer = 2, no connection.

## Function Prototype

```c
HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink **links, uint32_t *linkNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| srcRank | Input | Source rank ID. |
| dstRank | Input | Destination rank ID. |
| links | Output | Communication link list.<br>For the definition of the CommLink type, see [CommLink](../../datatype_definition/CommLink.md). |
| linkNum | Output | Number of communication links. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The returned memory is managed internally by the library, and the caller must not release it.
- The returned data should be copied promptly. Repeated calls in the same communicator may invalidate the previous result.

## Example

```c
HcclComm comm;
CommLink *links;
uint32_t linkNum;
uint32_t netlayer = 0;
// Query the link between rank0 and rank1 within the server.
HcclRankGraphGetLinks(comm, netlayer, 0, 1, &links, &linkNum);
```
