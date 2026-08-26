# HcclEngineCtxGet

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:29:52.110Z pushedAt=2026-08-17T07:03:42.213Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Specifies a communicator and a communication engine, and obtains the corresponding communication engine context by using the communication engine context tag.

## Function Prototype

```c
HcclResult HcclEngineCtxGet(HcclComm comm, const char *ctxTag, CommEngine engine, void **ctx, uint64_t *size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| ctxTag | Input | Communication engine context tag. The maximum character length is HCCL_RES_TAG_MAX_LEN.<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| engine | Input | Communication engine type. |
| ctx | Output | Communication engine context handle. |
| size | Output | Memory size corresponding to the communication engine context. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm;
uint64_t size = 0;
void *ctx = nullptr;
const char *ctxTag = "ctxTag";
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxGet(comm, ctxTag, engine, &ctx, &size);

```
