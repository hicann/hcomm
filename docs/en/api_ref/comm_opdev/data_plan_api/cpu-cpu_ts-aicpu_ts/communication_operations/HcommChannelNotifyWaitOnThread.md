# HcommChannelNotifyWaitOnThread

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T10:23:23.666Z pushedAt=2026-08-18T08:28:29.631Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Waits for a synchronization signal and blocks until the Notify on the specified channel is complete.

## Function Prototype

```c
int32_t HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle. For the CPU engine RoCE scenario of Ascend 950PR/Ascend 950DT, this parameter has no effect and can be set to 0. For the CPU_TS/AICPU_TS scenario, it is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channels obtained through the [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. For channel constraints, see Constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| localNotifyIdx | Input | Local Notify index.<br>Value range: [0, notifyNum).<br>notifyNum is the notifyNum in the channelDescs parameter passed to the [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. |
| timeOut | Input | Timeout duration, in seconds.<br>  - 0: indicates waiting indefinitely.<br>  - >0: the configured specific timeout duration.<br>Note: For the CPU engine RoCE scenario of Ascend 950PR/Ascend 950DT, a timeout duration greater than 0 must be configured. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

- This API must be used together with [HcommChannelNotifyRecordOnThread](HcommChannelNotifyRecordOnThread.md).
- For Ascend 950PR/Ascend 950DT, the API supports call on the device side in the AICPU_TS scenario, and also supports call on the host CPU side in the CPU engine RoCE scenario.
- For the CPU engine RoCE scenario on Ascend 950PR/Ascend 950DT, when calling [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) to allocate the input parameter `channel`, set `engine = COMM_ENGINE_CPU` and `channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE`. This API is not supported for channels using protocols such as URMA or UBC.
- When called on the host CPU side, the `thread` parameter has no effect and can be set to 0.
- For the CPU engine RoCE scenario on Ascend 950PR/Ascend 950DT, `localNotifyIdx` must be less than the number of Notify resources on the local communication channel, and `notifyNum` must be greater than 0 when the communication channel is created. `timeOut` must be greater than 0.

## Example

```c
// Allocate communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
// Configure this when Ascend 950PR/Ascend 950DT is used.
// CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// For Ascend 950PR/Ascend 950DT, call the following APIs on the device side.

// Notify the remote end.
HcommChannelNotifyRecordOnThread(thread, channel, 0);

// Perform data plane operations.
// ...

// Wait for the remote end to notify the local end.
uint32_t notifyTimeout = 1800;
HcommChannelNotifyWaitOnThread(thread, channel, 0, notifyTimeout);
```
