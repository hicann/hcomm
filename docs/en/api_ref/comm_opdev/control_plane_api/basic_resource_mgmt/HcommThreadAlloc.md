# HcommThreadAlloc

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:16:52.958Z pushedAt=2026-08-17T03:20:40.926Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Allocates communication threads. Currently, the AI CPU, AI CPU+TS, HOST CPU, and HOST CPU+TS communication engines are supported. Note: If the communication engine is AI CPU+TS, an additional kernel delivery must be performed before the AI CPU side can use this communication thread.

## Function Prototype

```c
HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t *notifyNumPerThread, ThreadHandle* threads)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| threadNum | Input | Number of communication threads. The maximum number of threads that can be allocated in each call of this API is 200. |
| notifyNumPerThread | Input | Number of synchronization resources (Notify) in each communication thread. The maximum number of Notify resources that can be allocated in each call of this API for each communication thread is 64. |
| threads | Output | Returned communication thread handles. A ThreadHandle array of the threadNum size must be passed in.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

HcommResult: The API returns 0 on success and other values on failure.

## Constraints

1. The threads allocated by calling this API must be released by calling [HcommThreadFree](HcommThreadFree.md) later. Before calling HcommThreadAlloc to allocate threads of communication engines such as AICPU_TS or CPU_TS, you must first call aclrtSetdevice on the same thread to specify the deviceId.

2. This API does not support the COMM_ENGINE_AIV and COMM_ENGINE_CCU communication engines.

## Example

```c
ThreadHandle thread[2];
// Allocate two streams, with three Notify resources per stream.
const uint32_t notifyNumPerThread[2] = {3, 3};
HcommResult ret =  HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, notifyNumPerThread, thread);
```
