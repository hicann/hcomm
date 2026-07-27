# HcclThreadAcquireWithConfig

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

基于通信域和线程配置获取通信线程，并为每条通信线程分配指定数量的同步资源（Notify）。相关概念可参见[通信算子开发指南-并发模型](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md)章节。

> [!NOTE]说明
> 与[HcclThreadAcquire](./HcclThreadAcquire.md)相比，该接口支持通过ThreadConfig结构体逐线程配置同步资源（Notify）数量，适用于不同线程需要不同Notify数量的场景。

## 函数原型

```c
HcclResult HcclThreadAcquireWithConfig(HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig *config, ThreadHandle *threads)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| engine | 输入 | 通信引擎类型。<br>CommEngine类型的定义可参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| threadNum | 输入 | 通信线程数量。一个通信域内最多申请40条流。 |
| type | 输入 | 线程类型。当前支持THREAD_TYPE_TS。 |
| config | 输入 | 每线程的配置信息，数组长度须与threadNum一致。调用前须使用ThreadConfigInit初始化。<br>ThreadConfig结构体定义如下：<br>```c typedef struct { CommAbiHeader header; uint16_t notifyNumPerThread; uint8_t reserved[14]; } ThreadConfig; ```<br>其中notifyNumPerThread表示每个通信线程中的同步资源（Notify）数量，取值范围0~65535。一个通信域内所有线程的同步资源总量不超过65536个。 |
| threads | 输出 | 返回的通信线程句柄。需传入threadNum大小的ThreadHandle类型数组。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口返回值说明如下：

| 返回值 | 描述 |
| --- | --- |
| HCCL_SUCCESS | 成功获取通信线程资源。 |
| HCCL_E_PTR | 传入的comm、threads或config为空指针。 |
| HCCL_E_PARA | 参数错误，可能原因：type无效、engine无效、threadNum为0、config未使用ThreadConfigInit初始化、engine不支持（CPU_TS/AICPU_TS/AIV/CCU）。 |
| HCCL_E_INTERNAL | 内部错误。 |

## 约束说明

1. config数组必须使用ThreadConfigInit函数进行初始化，否则接口返回参数错误。

2. 返回的通信线程与同步资源由库内管理，调用者严禁释放。

3. 当前各产品形态支持的CommEngine范围：

  <!-- npu="950" id6 -->
  - Ascend 950PR/Ascend 950DT：
    - COMM_ENGINE_CPU
    - COMM_ENGINE_AICPU
  <!-- end id6 -->

  <!-- npu="A3" id7 -->
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：
    - COMM_ENGINE_CPU
    - COMM_ENGINE_AICPU
  <!-- end id7 -->

  <!-- npu="910b" id8 -->
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：
    - COMM_ENGINE_CPU
    - COMM_ENGINE_AICPU
  <!-- end id8 -->

4. 该接口不支持COMM_ENGINE_CPU_TS和COMM_ENGINE_AICPU_TS通信引擎，如需TS类型线程，请使用COMM_ENGINE_CPU或COMM_ENGINE_AICPU引擎配合THREAD_TYPE_TS线程类型。

5. 该接口不支持COMM_ENGINE_AIV和COMM_ENGINE_CCU两种通信引擎。

## 调用示例

创建线程资源示例如下：

```c
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

不同线程配置不同Notify数量的示例如下：

```c
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
