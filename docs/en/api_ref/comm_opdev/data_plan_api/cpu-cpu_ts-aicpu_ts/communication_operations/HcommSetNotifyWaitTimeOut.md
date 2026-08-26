# HcommSetNotifyWaitTimeOut

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T10:28:20.716Z pushedAt=2026-08-18T09:15:39.884Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sets the default timeout for notification waiting. This timeout applies to the subsequent calls to [HcommChannelNotifyWaitOnThreadWithDefaultTimeout](./HcommChannelNotifyWaitOnThreadWithDefaultTimeout.md) and [HcommThreadNotifyWaitOnThreadWithDefaultTimeout](../local_operations/HcommThreadNotifyWaitOnThreadWithDefaultTimeout.md).

## Function Prototype

```c
int32_t HcommSetNotifyWaitTimeOut(float timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| timeOut | Input | Notification wait timeout.<br>Unit: second.<br>Value description:<br>  - 0: waits indefinitely without timeout.<br>  - >0: specific timeout (in seconds).<br>Default value: 1836 seconds (about 30 minutes).<br>Currently, only integers are supported. |

## Return Value

int32_t: returns 0 on success and a non-zero value on failure.

## Timeout Mechanism

1. **Default timeout**
   - If this API is not called, the default timeout is 1836 seconds (about 30 minutes).

2. **Conditions for the timeout to take effect**
   - The timeout set by this API applies to subsequent `*WithDefaultTimeout` APIs.
   - The timeout set by calling `HcommChannelNotifyWaitOnThread` or `HcommThreadNotifyWaitOnThread` is not affected by this API

3. **Special handling in non-AI CPU mode**
   - In non-AI CPU mode, if this API is not called, the default timeout is used.
   - The system automatically adds a 50-second offset to the default value as a safety buffer.

## Constraints

- The timeout set by this API takes effect only for the `*WithDefaultTimeout` series of APIs.
- `HcommChannelNotifyWaitOnThread` and `HcommThreadNotifyWaitOnThread`, which accept a manually specified timeout, are not affected by this API.
- It is recommended that you set the default timeout before starting data plane operations.
- Currently, `timeOut` supports only integer values, not decimal values.

## Example

```c
// 1. Set the default timeout (for example, 10 minutes).
float defaultTimeOut = 600;  // 600 seconds = 10 minutes
HcommSetNotifyWaitTimeOut(defaultTimeOut);

// 2. Allocate communication thread resources.
CommEngine engine = COMM_ENGINE_AICPU_TS;
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

// 6. Wait for the notification using the default timeout (no need to manually pass the timeout parameter).
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, 0);

// Or use thread notification wait.
HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, 0);
```

## Related APIs

- [HcommChannelNotifyWaitOnThreadWithDefaultTimeout](./HcommChannelNotifyWaitOnThreadWithDefaultTimeout.md): Waits for a channel notification using the default timeout.
- [HcommThreadNotifyWaitOnThreadWithDefaultTimeout](../local_operations/HcommThreadNotifyWaitOnThreadWithDefaultTimeout.md): Waits for a thread notification using the default timeout.
- [HcommChannelNotifyWaitOnThread](./HcommChannelNotifyWaitOnThread.md): Waits for a channel notification with a manually specified timeout.
- [HcommThreadNotifyWaitOnThread](../local_operations/HcommThreadNotifyWaitOnThread.md): Waits for a thread notification with a manually specified timeout.
