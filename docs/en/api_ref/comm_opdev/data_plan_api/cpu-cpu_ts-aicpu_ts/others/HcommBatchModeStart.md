# HcommBatchModeStart

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-14T10:41:30.420Z pushedAt=2026-08-18T11:39:42.371Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Enables batch mode. All data plane API calls (such as HcommLocalCopy and HcommWrite) between HcommBatchModeStart and HcommBatchModeEnd are cached and not executed immediately. All operations are submitted and executed together when HcommBatchModeEnd is called.

## Function Prototype

```c
int32_t HcommBatchModeStart(const char *batchTag)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| batchTag | Input | Batch task identifier (optional). If NULL or an empty string is passed, the task is a temporary batch task and is not cached after execution. If a non-empty string is passed, it is used to identify and manage subsequent batch tasks.<br>Note that in the AI CPU + TS communication engine scenario, task cache management based on a non-empty identifier is not yet fully supported. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

1. HcommBatchModeStart and HcommBatchModeEnd must be called in pairs and executed in the same thread.
2. Operations cached in batch mode are actually executed only after HcommBatchModeEnd is called.
3. Only Ascend 950PR/Ascend 950DT support batch mode and immediate execution mode (without calling the batch APIs). Other products must use batch mode.

## Example

```c
char *tag = "";
// Start batch mode (temporary batch task).
HcommBatchModeStart(tag);

// Calling data plane APIs in batch mode does not execute them immediately.
// ...

// End batch mode and trigger execution.
HcommBatchModeEnd(tag);
```
