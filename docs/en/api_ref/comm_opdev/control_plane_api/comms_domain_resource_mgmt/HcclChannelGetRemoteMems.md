# HcclChannelGetRemoteMems

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:24:14.940Z pushedAt=2026-08-17T06:39:44.300Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Obtains the remote memory information exchanged in a communication channel. The returned remote memory list contains all remote memory (including the default HcclBuffer of the communicator), not only the memory registered by users.

## Function Prototype

```c
HcclResult HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t *memNum, CommMem **remoteMems, char ***memTags);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| channel | Input | Communication channel handle.<br>For details about the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md). |
| memNum | Output | Number of memory entries. |
| remoteMems | Output | Remote memory list, which contains all remote memory (including the default HcclBuffer of the communicator).<br>For details about the CommMem type, see [CommMem](../../datatype_definition/CommMem.md). |
| memTags | Output | String list of remote memory. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

This API is supported only on the AIV engine of Ascend 950PR/Ascend 950DT.

## Example

```c
uint32_t channelNum = 1;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);

// Omit the channelDesc configuration.

HcclComm comm;
CommEngine engine = CommEngine::COMM_ENGINE_AIV;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDesc.data(), channelNum, channels.data());

uint32_t memNum = 0;
CommMem* remoteMems = nullptr;
char** memTags = nullptr;
HcclChannelGetRemoteMems(comm, channels[0], &memNum, &remoteMems, &memTags);
```
