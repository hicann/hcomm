# HcclEngineCtxCreate

<!-- md-trans-meta sourceCommit=5d2b6a1c1d3ecb2aedf23e593d18044c239c2ddc translatedAt=2026-08-14T09:27:15.719Z pushedAt=2026-08-17T07:00:06.235Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Creates a communication engine context with a specific tag for a specified communicator and communication engine.

A communication engine context is a block of memory that can be used by the data plane of the communication engine to store information such as resource handles or parameters required for executing operators. Once created, it can be obtained and used repeatedly. By specifying a communicator and a communication engine type, a communication engine tag can index a communication engine context.

## Function Prototype

```c
HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine, uint64_t size, void **ctx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| ctxTag | Input | Communication engine context tag. The maximum character length is HCCL_RES_TAG_MAX_LEN.<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| engine | Input | Communication engine type.<br>For the definition of CommEngine, see [CommEngine](../../datatype_definition/CommEngine.md). |
| size | Input | Size of the ctx memory, in bytes.<br>size cannot be 0. |
| ctx | Output | Communication engine context. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
// Communicator handle
HcclComm comm;
// Create a 16B communication engine context.
uint64_t size = 16;
void *ctx = nullptr;
string ctxTag = "ctxTag";
CommEngine engine = COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
if (ret != HCCL_SUCCESS) {
    // Error handling
}
```
