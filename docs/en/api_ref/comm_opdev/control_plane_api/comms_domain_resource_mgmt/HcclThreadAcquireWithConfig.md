# HcclThreadAcquireWithConfig

<!-- md-trans-meta sourceCommit=cdbf2128913fa6ce34016febb6934fd27ace94e9 translatedAt=2026-08-14T09:31:27.495Z pushedAt=2026-08-17T07:38:35.789Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains communication threads based on the communicator and thread configuration, and allocates a specified number of synchronization resources (Notify) to each communication thread. For related concepts, see [Communication Operator Development Guide - Concurrency Model](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md).

> [!NOTE] Note
> Compared with [HcclThreadAcquire](./HcclThreadAcquire.md), this API supports configuring the number of synchronization resources (Notify) for each thread through the ThreadConfig structure, which is suitable for scenarios where different threads require different numbers of Notify resources.

## Function Prototype

```c
HcclResult HcclThreadAcquireWithConfig(HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig *config, ThreadHandle *threads)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| threadNum | Input | Number of communication threads. |
| type | Input | Thread type.<br>For the definition of the ThreadType type, see [ThreadType](../../datatype_definition/ThreadType.md). |
| config | Input | Array of thread configuration information. The array length must be the same as threadNum. Before calling this API, call [ThreadConfigInit](./ThreadConfigInit.md) to initialize the parameters.<br>For the definition of the ThreadConfig type, see [ThreadHandle](../../datatype_definition/ThreadConfig.md). |
| threads | Output | Returned communication thread handles. Pass in a ThreadHandle array of size threadNum.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API return values are described as follows:

## Constraints

1. The returned communication thread and synchronization resources are managed by the library. Callers must not release them.

2. The CommEngine ranges supported by each product form are as follows:

   - Ascend 950PR/Ascend 950DT:
     - COMM_ENGINE_CPU
     - COMM_ENGINE_AICPU

   - Atlas A3 training products/Atlas A3 inference products:
     - COMM_ENGINE_CPU
     - COMM_ENGINE_AICPU

   - Atlas A2 training products/Atlas A2 inference products:
     - COMM_ENGINE_CPU
     - COMM_ENGINE_AICPU

3. This API does not support the COMM_ENGINE_AIV, COMM_ENGINE_CCU, COMM_ENGINE_CPU_TS, and COMM_ENGINE_AICPU_TS communication engines.

  > To create a TS-type thread, pass the COMM_ENGINE_CPU or COMM_ENGINE_AICPU communication engine and set `type` to THREAD_TYPE_TS.

## Example

The following shows an example of creating thread resources:

```c
// Communicator handle
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU;
uint32_t threadNum = 5;
ThreadConfig configs[5];
ThreadConfigInit(configs, threadNum);
for (uint32_t i = 0; i < threadNum; i++) {
    configs[i].notifyNumPerThread = 2;
}
ThreadHandle threads[5];
HcclThreadAcquireWithConfig(comm, engine, threadNum, THREAD_TYPE_TS, configs, threads);
```

The following shows an example of configuring different numbers of Notify resources for different threads:

```c
// Communicator handle
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU;
uint32_t threadNum = 3;
ThreadConfig configs[3];
ThreadConfigInit(configs, threadNum);
configs[0].notifyNumPerThread = 2;
configs[1].notifyNumPerThread = 4;
configs[2].notifyNumPerThread = 1;
ThreadHandle threads[3];
HcclThreadAcquireWithConfig(comm, engine, threadNum, THREAD_TYPE_TS, configs, threads);
```
