# HcommThreadResAcquireTimeOut

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T10:28:30.494Z pushedAt=2026-08-18T09:23:30.362Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sets the timeout period for thread resource acquisition. This timeout period applies to the queue-full waiting scenario of Remote Transport Sequence Queue (RTSQ). When the RTSQ queue is full, the system waits until space becomes available or a timeout occurs.

## Function Prototype

```c
int32_t HcommThreadResAcquireTimeOut(float timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| timeOut | Input | Timeout period for waiting when the RTSQ queue is full.<br>Unit: second.<br>Value description:<br>  - 0: Indicates that the timeout never occurs. The system waits until space becomes available in RTSQ.<br>  - >0: Specific timeout period (in seconds).<br>Default value: 1856 seconds.<br>Currently, only integers are supported. |

## Return Value

int32_t: This API returns 0 on success and a non-zero value on failure.

## Timeout Mechanism

1. **Timeout value semantics**
   - When the RTSQ queue is full, the thread blocks and waits until space becomes available.
   - If a timeout period is set, an error is reported when no space becomes available after the timeout period expires.
   - Setting the value to 0 means no timeout, and the thread waits indefinitely.

2. **Default value**
   - If this API is not called to set a timeout period, the default timeout period is 1856 seconds (about 30 minutes).

3. **Setting scope**
   - It applies to all RTSQ resources acquired through threads.
   - After the setting takes effect, it affects all subsequent resource acquisition operations.

## Scenarios

This API is mainly used in the following scenarios:

- **RTSQ queue-full waiting**: When the send buffer is full, wait for available space.
- **Flow control mechanism**: Work with send operations to implement the backpressure mechanism.
- **Resource acquisition**: Ensure that resources are available before performing subsequent operations.

## Constraints

- The timeout period set by this API affects the resource acquisition operations of all threads.
- It is recommended that you set an appropriate timeout value during initialization or before allocating resources.
- If the timeout value is set to 0, the thread waits indefinitely, which may cause deadlock. Use this value with caution.
- Currently, only integers are supported for `timeOut`. Decimal values are not supported.

## Example

```c
// 1. Set the RTSQ queue-full wait timeout period (for example, 10 minutes).
float timeout = 600;  // 600 seconds = 10 minutes
HcommThreadResAcquireTimeOut(timeout);

// Or set it to never time out (0 means never time out).
float timeoutNever = 0;
HcommThreadResAcquireTimeOut(timeoutNever);

// 2. Allocate the communication thread resource.
CommEngine engine = COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// 3. Perform data plane operations (when RTSQ is full, wait for available space or time out).
// ... Perform data sending operations ...
```

## Related APIs

- [HcommSetNotifyWaitTimeOut](./HcommSetNotifyWaitTimeOut.md): Sets the notification wait timeout period.
