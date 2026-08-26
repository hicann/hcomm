# HcommThreadFree

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:16:48.086Z pushedAt=2026-08-17T03:22:24.621Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Releases threads allocated by calling the HcommThreadAlloc API.

## Function Prototype

```c
HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| threads | Input | Communication thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |
| threadNum | Input | Number of communication threads. |

## Return Value

HcommResult: The API returns 0 on success and other values on failure.

## Constraints

Only threads allocated by the HcommThreadAlloc API can be released.

## Example

```c
ThreadHandle thread[2];
HcommResult ret =  HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, 3, thread);
HcommResult ret =  HcommThreadFree(thread, 2);
```
