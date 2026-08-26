# HcommThreadNotifyWaitOnThread

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T10:39:09.279Z pushedAt=2026-08-18T11:12:45.004Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Waits for a synchronization signal. This API blocks and waits for the thread to run until the specified Notify is recorded.

## Function Prototype

```c
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Thread handle, which is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| notifyIdx | Input | Index of the Notify to wait for.<br>Value range: [0, the value of the notifyNumPerThread parameter passed to the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API). |
| timeOut | Input | Timeout duration, in seconds.<br>  - 0: waits indefinitely.<br>  - >0: the configured timeout duration.<br> |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

This API must be used together with [HcommThreadNotifyRecordOnThread](HcommThreadNotifyRecordOnThread.md).

On Ascend 950PR/Ascend 950DT, this API can be called only in AICPU_TS mode on the device side.

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream streams[2];
ThreadHandle threads[2];
// Allocate two streams, with two Notify resources per stream.
aclrtCreateStream(&streams[0]);
aclrtCreateStream(&streams[1]);
HcclResult result = HcclThreadAcquireWithStream(comm, engine, streams[0], 2, &threads[0]);
result = HcclThreadAcquireWithStream(comm, engine, streams[1], 2, &threads[1]);
uint32_t notifyIdx = 0;
// Send the synchronization signal.
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
uint32_t timeout = 1;
// Wait for the synchronization signal.
HcommThreadNotifyWaitOnThread(threads[1], notifyIdx, timeout);
```

On Ascend 950PR/Ascend 950DT, this function must be compiled for use on the device side:

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
ThreadHandle threads[2];
uint32_t notifyNumPerThread = 2;
HcclThreadAcquire(comm, engine, 1, notifyNumPerThread, &threads[0]);
HcclThreadAcquire(comm, engine, 1, notifyNumPerThread, &threads[1]);

// Allocate other resources.
// Copy parameters and launch the kernel.

// Orchestrate the algorithm on the device side.
uint32_t notifyIdx = 0;
// Send the synchronization signal.
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
uint32_t timeout = 1;
// Wait for the synchronization signal.
HcommThreadNotifyWaitOnThread(threads[1], notifyIdx, timeout);
```
