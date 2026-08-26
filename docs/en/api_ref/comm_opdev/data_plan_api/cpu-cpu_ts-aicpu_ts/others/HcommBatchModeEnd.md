# HcommBatchModeEnd

<!-- md-trans-meta sourceCommit=7ff807bedd6173de4c7cb9ba16dadd5138b23868 translatedAt=2026-08-14T10:41:18.498Z pushedAt=2026-08-18T11:37:23.474Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Submits and triggers the execution of all operations cached in batch mode. All data plane API calls made between HcommBatchModeStart and HcommBatchModeEnd are executed at this point.

## Function Prototype

```c
int32_t HcommBatchModeEnd(const char *batchTag)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| batchTag | Input | Batch task identifier, which must be consistent with the batchTag passed to HcommBatchModeStart. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

HcommBatchModeStart and HcommBatchModeEnd must be called in pairs and executed in the same thread.

Only Ascend 950PR and Ascend 950DT support batch mode and immediate execution mode (without calling batch APIs). Other products must use batch mode.

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
