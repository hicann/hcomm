# HcclChannelAcquire

<!-- md-trans-meta sourceCommit=6d41cf8dd0b097be1993a10deb9a75d9b04739d9 translatedAt=2026-08-14T09:23:22.271Z pushedAt=2026-08-17T06:26:33.749Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains multiple communication channels based on a communicator. If the corresponding communication channels do not exist in the communicator, they are created directly.

Whether a channel is reused is determined by the unique channel identifier consisting of commId + engine + remoterank + channelProtocol.

## Function Prototype

```c
HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| engine | Input | Communication engine type.<br>For details about the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| channelDescs | Input | Communication channel description list. The list length is channelNum.<br>For details about the HcclChannelDesc type, see [HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md). The list is obtained by calling [HcclRankGraphGetLinks](../topo_info_query/HcclRankGraphGetLinks.md). |
| channelNum | Input | Number of communication channels. The value range of channelNum is (0, 1024 * 1024]. |
| channels | Output | Communication channel handle list. The list length is channelNum. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

1. When CommEngine is set to CCU, external configuration of NotifyNum is not supported, and eight CCU Notify resources are allocated by default.

2. When CommEngine is set to CCU, exchanging additional custom memory is not supported. Only the HcclBuffer of the communicator can be exchanged.

3. The communication protocols supported by each CommEngine depend on the chip model, as described below:

  For Ascend 950PR/Ascend 950DT, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_CPU
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_UBOE
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_AIV
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    - COMM_PROTOCOL_UB_MEM
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_CCU
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP

  For Atlas A3 training products/Atlas A3 inference products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY

  For Atlas A2 training products/Atlas A2 inference products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_CPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY

## Example

Take batch communication channels as an example:

```c
// 1. Call HcclRankGraphGetLinks to obtain link information.
CommLink *linkList = nullptr;
uint32_t listSize;
CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));

// 2. Traverse each CommLink and fill in HcclChannelDesc.
uint32_t channelNum = listSize;
std::vector<HcclChannelDesc> channelDescVec(channelNum);
for (uint32_t idx = 0; idx < listSize; idx++) {
  HcclChannelDesc channelDesc;
  HcclChannelDescInit(&channelDesc, 1);
  channelDesc.remoteRank = rank;

  CommLink link = linkList[idx];

  //  Core mapping: extract endpoint information from CommLink.
  channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
  channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
  channelDesc.localEndpoint.loc    = link.srcEndpointDesc.loc;
  channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
  channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
  channelDesc.remoteEndpoint.loc   = link.dstEndpointDesc.loc;
  channelDesc.channelProtocol     = link.linkAttr.linkProtocol;
  channelDesc.notifyNum = 8; // Specify the number of Notify resources as required.

  channelDescVec[idx] = channelDesc;
}

// 3. Create channels in batches.
HcclComm comm;
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDescVec.data(), channelNum, channels.data());
```
