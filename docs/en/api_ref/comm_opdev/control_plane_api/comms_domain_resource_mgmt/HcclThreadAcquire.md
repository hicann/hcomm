# HcclThreadAcquire

<!-- md-trans-meta sourceCommit=cdbf2128913fa6ce34016febb6934fd27ace94e9 translatedAt=2026-08-14T09:31:10.526Z pushedAt=2026-08-17T07:32:06.139Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains communication threads based on a communicator and allocates a specified number of synchronization resources (Notify) to each communication thread. For related concepts, see the [Communication Operator Development Guide - Concurrency Model](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md) section.

## Function Prototype

```c
HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle *threads)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| threadNum | Input | Number of communication threads. |
| notifyNumPerThread | Input | Number of synchronization resources (Notify) in each communication thread. The value range is \[0, 65535\], and the specific limit is determined by the specific product. Configure a proper value based on the service scenario to avoid resource shortage or waste. |
| threads | Output | Returned communication thread handles. Pass in a ThreadHandle array of size threadNum.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

1. The returned communication threads and synchronization resources are managed by the library. Callers must not release them.

2. The CommEngine ranges supported by each product form are as follows:

   - Ascend 950PR/Ascend 950DT:
     - COMM_ENGINE_CPU_TS
     - COMM_ENGINE_AICPU_TS

   - Atlas A3 training products/Atlas A3 inference products:
     - COMM_ENGINE_CPU_TS
     - COMM_ENGINE_AICPU_TS

   - Atlas A2 training products/Atlas A2 inference products:
     - COMM_ENGINE_CPU_TS
     - COMM_ENGINE_AICPU_TS

3. This API does not support the COMM_ENGINE_AIV and COMM_ENGINE_CCU communication engines.

## Example

The following shows an example of creating thread resources:

```c
// Communicator handle
HcclComm comm;
// Allocate five AICPU_TS communication threads, each containing two Notify resources.
CommEngine engine = COMM_ENGINE_AICPU_TS;
ThreadHandle threads[5];
HcclThreadAcquire(comm, engine, 5, 2, threads);
```

The following shows an example of synchronizing host thread resources:

```c
// Allocate a host stream.
aclrtStream stream;
aclrtCreateStream(&stream);
// Create a CPU_TS thread based on the allocated stream.
ThreadHandle cpuThread;
HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &cpuThread);
// Schedule tasks.
// ...
// Synchronize the stream.
aclrtSynchronizeStream(stream);
```

The following shows an example of synchronizing AI CPU thread resources:

```c
// --Host-side call process--
// Allocate a host stream.
aclrtStream stream;
aclrtCreateStream(&stream);
// Create a CPU_TS thread based on the allocated stream.
ThreadHandle cpuThread;
HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &cpuThread);

// Create an AICPU_TS thread.
ThreadHandle aicpuThread;
HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 1, &aicpuThread);

// Export the created AICPU_TS thread as a thread available on the CPU.
ThreadHandle exportedCpuThread;
HcclThreadExportToCommEngine(comm, 1, &aicpuThread, COMM_ENGINE_CPU_TS, &exportedCpuThread);
// Export the created CPU thread as a thread available on the AI CPU.
ThreadHandle exportedAicpuThread;
HcclThreadExportToCommEngine(comm, 1, &cpuThread, COMM_ENGINE_AICPU_TS, &exportedAicpuThread);

// Send a synchronization signal.
HcommThreadNotifyRecordOnThread(cpuThread, exportedCpuThread, 0);
// Launch the kernel to bring exportedAicpuThread and aicpuThread to the AI CPU side.
// ...
uint32_t timeout = 1;
// Wait for the synchronization signal.
HcommThreadNotifyWaitOnThread(cpuThread, 0, timeout);

// --Device-side call process--
// Wait for the synchronization signal.
uint32_t timeout = 1;
HcommThreadNotifyWaitOnThread(aicpuThread, 0, timeout);
// Schedule tasks and dispatch them to aicpuThread.
// ...
// Send the synchronization signal.
HcommThreadNotifyRecordOnThread(aicpuThread, exportedAicpuThread, 0);

// --Host-side call process--
// Synchronize the stream.
aclrtSynchronizeStream(stream);
```
