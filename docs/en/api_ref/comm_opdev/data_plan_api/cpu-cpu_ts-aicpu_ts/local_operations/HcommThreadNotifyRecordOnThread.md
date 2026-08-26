# HcommThreadNotifyRecordOnThread

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-14T10:38:05.927Z pushedAt=2026-08-18T11:10:09.585Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sends a synchronization signal to another thread. It is mainly used in multi-thread synchronization wait scenarios.

## Function Prototype

```c
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](.././../../datatype_definition/ThreadHandle.md). |
| dstThread | Input | Target communication thread handle, which is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| dstNotifyIdx | Input | Target notify index.<br>Value range: [0, the value of the notifyNumPerThread parameter passed to [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)). |

## Return Value

int32_t: The API returns 0 on success and other values on failure.

## Constraints

This API must be used together with [HcommThreadNotifyWaitOnThread](HcommThreadNotifyWaitOnThread.md).

On Ascend 950PR/Ascend 950DT, this API can be called only on the device side in AICPU_TS mode.

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream streams[2];
ThreadHandle threads[2];
// Allocate two streams, each with two Notify resources.
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

// Device-side algorithm orchestration
uint32_t notifyIdx = 0;
// Send a synchronization signal.
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
uint32_t timeout = 1;
// Wait for a synchronization signal.
HcommThreadNotifyWaitOnThread(threads[1], notifyIdx, timeout);
```
