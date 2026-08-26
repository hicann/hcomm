# HcclChannelGetHcclBuffer

<!-- md-trans-meta sourceCommit=5d2b6a1c1d3ecb2aedf23e593d18044c239c2ddc translatedAt=2026-08-14T09:24:00.296Z pushedAt=2026-08-17T06:32:57.150Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the HCCL communication memory of the peer end of a specified channel.

## Function Prototype

```c
HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void **buffer, uint64_t *size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| channel | Input | Communication channel handle.<br>For details about the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md). |
| buffer | Output | HCCL communication memory address. |
| size | Output | HCCL communication memory size. The memory size is twice the value configured during communicator initialization or the value of the HCCL_BUFFSIZE environment variable, and defaults to 400 MB. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Set the device for the current thread.
aclrtSetDevice(0);

// Initialize the channel descriptor.
uint32_t channelNum = 1;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);

// The peer is Rank1.
channelDesc[0].remoteRank = 1;
channelDesc[0].channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
channelDesc[0].notifyNum = 3;

// Communicator handle.
HcclComm comm;
// Create channel resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDesc.data(), channelNum, channels.data());

// Obtain the address and size of the HCCL buffer of the peer end (Rank1).
void *remoteBufferAddr;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channels[0], &remoteBufferAddr, &remoteBufferSize);
```
