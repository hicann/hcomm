# HcclThreadAcquireWithStream

<!-- md-trans-meta sourceCommit=cdbf2128913fa6ce34016febb6934fd27ace94e9 translatedAt=2026-08-14T09:32:41.759Z pushedAt=2026-08-17T07:42:04.988Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains a communication thread based on the communicator and Runtime stream handle, and allocates a specified number of synchronization resources (Notify) for the communication thread. For related concepts, see [Communication Operator Development Guide - Concurrency Model](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md).

## Function Prototype

```c
HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| stream | Input | Stream handle. |
| notifyNum | Input | Number of synchronization signals. |
| thread | Output | Thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

This API supports only the COMM_ENGINE_CPU, COMM_ENGINE_CPU_TS, and COMM_ENGINE_CCU communication engines.

## Example

```c
// Communicator handle
HcclComm comm;
// Create a runtime stream.
aclrtStream stream;
aclrtCreateStream(&stream);
// Create a thread and allocate Notify resources on two threads.
ThreadHandle thread;
HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 2, &thread);
if (ret != HCCL_SUCCESS) {
    // Error handling
}

// Data plane operations
// ...

// Synchronize the stream.
aclrtSynchronizeStream(stream);
```
