# HcommChannelNotifyRecord

<!-- md-trans-meta sourceCommit=59b5f25111e74c4a6aea4eef8ee409ef5bf248f6 translatedAt=2026-08-14T10:20:39.713Z pushedAt=2026-08-18T08:13:31.303Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Sends a synchronization signal and records a Notify on the specified channel. This API is asynchronous and is mainly used in scenarios where both ends of a channel wait for synchronization.

## Function Prototype

```c
int32_t HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channel | Input | Communication channel handle, which is the channels obtained through the [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. For channel constraints, see Constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| remoteNotifyIdx | Input | Notify index at the other end of the communication channel.<br>Value range: [0, notifyNum).<br>notifyNum is the notifyNum in the channelDesc parameter passed to the [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

<!-- npu="950" id6 -->
- For Ascend 950PR/Ascend 950DT, this API can be called only on the host CPU, and cannot be called on the AI CPU side.
- When calling [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) to allocate the input parameter channel, you need to pass `engine = COMM_ENGINE_CPU` and `channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE`. Channels using protocols such as URMA and UBC do not support this API.
- `remoteNotifyIdx` must be less than the number of Notify resources on the other end of the communication channel, and `notifyNum` must be greater than 0 when the communication channel is created.
<!-- end id6 -->
- This API must be used together with [HcommChannelNotifyWait](HcommChannelNotifyWait.md).

## Example

```c
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
HcclComm comm;

// Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
channelDesc.localEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.notifyNum = 1;
// Omitted: Fill in other information in channelDesc.
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// The following APIs are called on the host CPU side.

// Notify the peer end.
HcommChannelNotifyRecord(channel, 0);

// Data plane operations
// ...

// Wait for the remote end to notify the local end.
uint32_t notifyTimeout = 1800;
HcommChannelNotifyWait(channel, 0, notifyTimeout);
```
