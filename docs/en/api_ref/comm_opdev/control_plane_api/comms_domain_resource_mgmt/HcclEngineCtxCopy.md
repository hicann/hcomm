# HcclEngineCtxCopy

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:26:59.715Z pushedAt=2026-08-17T06:56:35.430Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Specifies the communicator, communication engine, and communication engine context tag to copy host-side memory data to the corresponding communication engine context.

## Function Prototype

```c
HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char *ctxTag, const void *srcCtx, uint64_t size, uint64_t dstCtxOffset)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| engine | Input | Communication engine type. |
| ctxTag | Input | Communication engine context tag (maximum character length: HCCL_RES_TAG_MAX_LEN). |
| srcCtx | Input | Source memory address. |
| size | Input | Source memory size. |
| dstCtxOffset | Input | Address offset in the communication engine context to which data is copied. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Communicator handle
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
const char *ctxTag = "ctxTag";
void *resCtx = nullptr;  // Valid source ctx, which must point to allocated memory.
uint64_t size = 16; // Actual size to be copied.
uint64_t dstCtxOffset = 0; // When copying all data, set the offset to 0.
HcclResult ret = HcclEngineCtxCopy(comm, engine, ctxTag, resCtx, size, dstCtxOffset);
if (ret != HCCL_SUCCESS) {
    // Handle the error.
}
```
