# HcclEngineCtxDestroy

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T09:28:11.947Z pushedAt=2026-08-17T07:02:06.351Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Destroys the communication engine context corresponding to a specified communicator and communication engine using a specific tag.

## Function Prototype

```c
HcclResult HcclEngineCtxDestroy(HcclComm comm, const char *ctxTag, CommEngine engine)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| ctxTag | Input | Communication engine context tag (maximum character length: HCCL_RES_TAG_MAX_LEN). |
| engine | Input | Communication engine type. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm;
uint64_t size = 16;
void *ctx = nullptr;
string ctxTag = "ctxTag";
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
ret = HcclEngineCtxDestroy(comm, ctxTag, engine);
```
