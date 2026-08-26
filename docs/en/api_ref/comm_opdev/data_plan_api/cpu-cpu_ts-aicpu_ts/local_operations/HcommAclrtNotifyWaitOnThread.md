# HcommAclrtNotifyWaitOnThread

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T10:36:55.507Z pushedAt=2026-08-18T11:02:10.611Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Not supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Waits for a synchronization signal based on the Notify created through the ACL API. This API must be used in pair with [HcommAclrtNotifyRecordOnThread](HcommAclrtNotifyRecordOnThread.md).

## Function Prototype

```c
int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Thread handle, which is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| notifyId | Input | Synchronization signal ID, which is the notifyId obtained through the aclrtGetNotifyId API. |
| timeOut | Input | Timeout duration, in seconds.<br>  - 0: waits indefinitely.<br>  - >0: the configured specific timeout duration. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

None

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream streams[2];
ThreadHandle threads[2];
// Allocate 2 streams, with 2 Notify resources per stream.
aclrtCreateStream(&streams[0]);
aclrtCreateStream(&streams[1]);
HcclResult result = HcclThreadAcquireWithStream(comm, engine, streams[0], 2, &threads[0]);
result = HcclThreadAcquireWithStream(comm, engine, streams[1], 2, &threads[1]);
aclrtNotify notify;
uint64_t notifyId;
aclrtCreateNotify(&(notify), ACL_NOTIFY_DEFAULT);
aclrtGetNotifyId(notify, &(notifyId));
// Send the synchronization signal.
HcommAclrtNotifyRecordOnThread(threads[0], notifyId);
// Wait for the synchronization signal.
uint32_t timeout = 1;
HcommAclrtNotifyWaitOnThread(threads[1], notifyId, timeout);
```
