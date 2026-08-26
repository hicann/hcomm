# HcommChannelNotifyWaitOnThreadWithDefaultTimeout

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T10:23:52.679Z pushedAt=2026-08-18T08:37:37.118Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Waits for a synchronization signal and blocks until the Notify on the specified channel is complete. **Uses the default timeout set through HcommSetNotifyWaitTimeOut**, so you do not need to manually pass a timeout parameter.

## Function Prototype

```c
int32_t HcommChannelNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle. In the CPU engine RoCE scenario of Ascend 950PR/Ascend 950DT, this parameter has no effect and can be set to 0. In the CPU_TS/AICPU_TS scenario, it is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channels obtained through the [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. For channel constraints, see Constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| localNotifyIdx | Input | Local Notify index.<br>Value range: [0, notifyNum).<br>notifyNum is the notifyNum in the channelDescs parameter passed to the [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Default Timeout Description

- The default timeout is set through the [HcommSetNotifyWaitTimeOut](./HcommSetNotifyWaitTimeOut.md) API.
- If it is not set, in AICPU_TS mode, the default timeout on the device side is 1836 seconds.
- Setting it to 0 indicates waiting indefinitely.
- Setting it to a value greater than 0 indicates a specific timeout (in seconds).

## Constraints

- This API must be used together with [HcommChannelNotifyRecordOnThread](HcommChannelNotifyRecordOnThread.md).
- On Ascend 950PR/Ascend 950DT, this API can be called only on the device side in AICPU_TS mode.
- In AICPU_TS mode, before calling this API on the device side, if you need to set the timeout, call [HcommSetNotifyWaitTimeOut](./HcommSetNotifyWaitTimeOut.md). If the setting API is not called, the default timeout is 1836 seconds.
- For the AICPU_TS mode of Ascend 950PR/Ascend 950DT, `localNotifyIdx` must be smaller than the number of Notify resources of the local communication channel, and `notifyNum` must be greater than 0 when the communication channel is created.

## Example

```c
// 1. Set the default timeout (optional; if not set, the default value 1836 seconds is used).
uint32_t defaultTimeout = 1800;  // 30 minutes
HcommSetNotifyWaitTimeOut(defaultTimeout);

// 2. Allocate communication thread resources.
CommEngine engine = COMM_ENGINE_AICPU_TS;  // Configured for Ascend 950PR/Ascend 950DT.
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// 3. Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// 4. Record the notification on the device side.
HcommChannelNotifyRecordOnThread(thread, channel, 0);

// 5. Perform data plane operations.
// ...

// 6. Wait for the peer notification (use the default timeout, no need to pass it manually).
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, 0);
```
