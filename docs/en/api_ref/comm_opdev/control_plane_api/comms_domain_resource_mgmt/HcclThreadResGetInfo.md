# HcclThreadResGetInfo

<!-- md-trans-meta sourceCommit=5d2b6a1c1d3ecb2aedf23e593d18044c239c2ddc translatedAt=2026-08-14T09:33:41.949Z pushedAt=2026-08-17T07:45:47.388Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Obtains the underlying resources of a thread, such as streams.

## Function Prototype

```c
HcclResult HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void **info)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| thread | Input | Thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). A communication thread can be created by calling APIs such as [HcclThreadAcquire](./HcclThreadAcquire.md), [HcclThreadAcquireWithConfig](./HcclThreadAcquireWithConfig.md), and [HcclThreadAcquireWithStream](./HcclThreadAcquireWithStream.md). |
| resType | Input | Underlying resource type (such as stream).<br>For the definition of the ThreadResType type, see [ThreadResType](../../datatype_definition/ThreadResType.md). |
| infoLen | Input | Size of the target resource information. |
| info | Output | Output buffer for resource information. The return type is the corresponding resource type obtained. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

This API supports only obtaining the underlying stream resource of a communication thread (type: [ThreadResTypeStream](../../datatype_definition/ThreadResTypeStream.md)).

## Example

```c
// Communicator handle.
HcclComm comm;
ThreadHandle thread;          // Handle of the thread created by HcclThreadAcquire.
ThreadResTypeStream stream;   // The info buffer must be aligned by resource type and writable.
uint32_t size = sizeof(ThreadResTypeStream);  // Must be equal to the target type size.
HcclResult ret = HcclThreadResGetInfo(comm, thread, THREAD_RES_TYPE_STREAM, size, &stream);
if (ret != HCCL_SUCCESS) {
    // Error handling.
}
// Use the stream resource.
// ...
```
