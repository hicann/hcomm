# HcclCommInitAll

## 产品支持情况

<cann-filter npu-type="950">

- Ascend 950PR/Ascend 950DT：支持</cann-filter>
<cann-filter npu-type="A3">
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持</cann-filter>
<cann-filter npu-type="910b">
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持</cann-filter>
<cann-filter npu-type="310p">
- Atlas 推理系列产品：支持</cann-filter>
<cann-filter npu-type="910">
- Atlas 训练系列产品：支持</cann-filter>

<cann-filter npu-type="910b,310P">

> [!NOTE]说明
<cann-filter npu-type="910b">
> - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。</cann-filter>
<cann-filter npu-type="310p">
> - 针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。</cann-filter>
</cann-filter>

## 功能说明

单机通信场景中，通过一个进程统一创建多张卡的通信域（其中一张卡对应一个线程）。在初始化通信域的过程中，devices\[0\]作为root rank自动收集集群信息。

## 函数原型

```c
HcclResult HcclCommInitAll(uint32_t ndev, int32_t*  devices, HcclComm* comms)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ndev | 输入 | 通信域内的device个数。 |
| devices | 输入 | 通信域中的device列表，其值为device的逻辑ID，可通过npu-smi info -m命令查询，HCCL按照devices中设置的顺序创建通信域。<br>需要注意，输入的devices列表中不能包含重复的device ID。 |
| comms | 输出 | 生成的通信域句柄数组，其大小为：ndev * sizeof(HcclComm)。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 该接口仅支持单机通信场景使用，不支持多机通信场景。
- 多线程调用通信操作API时（例如HcclAllReduce等），用户需确保不同线程中调用通信操作API的前后时间差不超过环境变量[HCCL_CONNECT_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.1.0-beta.2/docs/zh/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md)的时间，避免建链超时。
- 不支持一张卡同时调用多个通信操作API。

## 调用示例

```c
uint32_t rankSize = 2;
int32_t devices[rankSize] = {0, 1};
HcclComm comms[rankSize];
// 初始化通信域
HcclCommInitAll(rankSize, devices, comms);
// 销毁通信域
for (uint32_t i = 0; i &lt; rankSize; i++) {
    HcclCommDestroy(comms[i]);
}
```
