# HcommWriteWithNotifyNbiOnThread

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-14T10:33:53.654Z pushedAt=2026-08-18T10:53:17.162Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Writes data to the specified memory on the channel, writes the memory data of length `len` in `src` to the memory area of the same length pointed to by `dst`, and sends a synchronization signal to the node where `dst` resides. The API caller is the node where `src` resides. This API is non-blocking.

## Function Prototype

```c
int32_t HcommWriteWithNotifyNbiOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len, uint32_t remoteNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle. It is not used currently and can be set to 0. |
| channel | Input | Communication channel handle, which is the channels obtained through [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md). For constraints on the channel, see Constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address, which is the local memory address obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| len | Input | Data length (in bytes). It must be greater than 0. |
| remoteNotifyIdx | Input | Notify index at the other end of the communication channel.<br>Value range: [0, notifyNum).<br>notifyNum is the notifyNum in the channelDesc parameter passed to [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md). |

## Return Value

int32_t: The API returns 0 on success and other values on failure.

## Constraints

- For Ascend 950PR/Ascend 950DT, this API can be called only on the host CPU. When calling [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) to allocate the input parameter `channel`, pass `engine = COMM_ENGINE_CPU` and `channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE`. This API is not supported on channels using protocols such as URMA/UBC.
- When called on the host CPU, the `thread` parameter has no effect and can be set to 0.
- `[src, src + len)` must fall within the memory range registered or obtained on the local end, and `[dst, dst + len)` must fall within the memory range imported or obtained on the remote end.
- `remoteNotifyIdx` must be smaller than the number of Notify resources on the other end of the communication channel, and `notifyNum` must be greater than 0 when the communication channel is created.
- This API must be used together with [HcommChannelNotifyWaitOnThread](HcommChannelNotifyWaitOnThread.md). The write target end should call this API to wait for the synchronization signal.
- A successful return from this API only indicates that the write request with notification has been submitted successfully. To confirm that the submitted write operations on the local channel are complete, the caller should call [HcommChannelFenceOnThread](HcommChannelFenceOnThread.md) to wait for the submitted write operations on the channel to complete.

## Example

```c
// Omitted: Create the communicator handle comm.

// Allocate communication channel resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclResult result = HcclChannelDescInit(&channelDesc, channelNum);
channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
channelDesc.localEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.notifyNum = 1;
// Omitted: Fill in other information in channelDesc.
ChannelHandle channel;
result = HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// Obtain the local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
result = HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// Obtain the remote communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
result = HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// Write the content of the local memory to the remote memory and notify the remote end.
uint32_t rmtNotifyIdx = 0;
int32_t ret = HcommWriteWithNotifyNbiOnThread(0, channel, remoteBuffer, localBuffer, len, rmtNotifyIdx);

// Wait for the submitted write operations on the communication channel to complete.
ret = HcommChannelFenceOnThread(0, channel);
```

The following code is called on the write target end to wait for the synchronization signal sent by the write initiator.

```c
// Omitted: resource creation

// Use the local Notify index to wait for the synchronization signal sent by the write initiator.
uint32_t lclNotifyIdx = 0;
uint32_t notifyTimeout = 0;
int32_t ret = HcommChannelNotifyWaitOnThread(0, channel, lclNotifyIdx, notifyTimeout);
```
