# HcclGroupStart

## 产品支持情况

<cann-filter npu-type="950">

- Ascend 950PR/Ascend 950DT：不支持</cann-filter>
<cann-filter npu-type="A3">
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持</cann-filter>
<cann-filter npu-type="910b">
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持</cann-filter>
<cann-filter npu-type="310p">
- Atlas 推理系列产品：不支持</cann-filter>
<cann-filter npu-type="910">
- Atlas 训练系列产品：不支持</cann-filter>

<cann-filter npu-type="910b">

> [!NOTE]说明
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。
</cann-filter>

## 功能说明

开始一个组调用。

在HcclGroupStart和HcclGroupEnd之间调用多个函数，作为一个整体执行，组调用支持以下三种场景：

- 单进程多线程管理NPU：支持调用通信域管理接口[HcclCommInitClusterInfo](HcclCommInitClusterInfo.md)、[HcclCommInitClusterInfoConfig](HcclCommInitClusterInfoConfig.md)、[HcclCommInitRootInfo](HcclCommInitRootInfo.md)、[HcclCommInitRootInfoConfig](HcclCommInitRootInfoConfig.md)  、[HcclCommDestroy](HcclCommDestroy.md)。
- 合并多个集合通信。
- 合并多个点对点通信。

## 函数原型

```c
HcclResult HcclGroupStart()
```

## 参数说明

无

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 仅支持在单机环境中使用组调用接口进行通信域管理。
- 在一个组调用中，通信域管理、集合通信、点对点通信类型的接口不可混用。
- 合并多个点对点通信时，不支持调用[HcclBatchSendRecv](https://gitcode.com/cann/hccl/blob/9.1.0-beta.3/docs/zh/api_ref/comm_op_interface/HcclBatchSendRecv.md)接口。
- HcclGroupStart必须和HcclGroupEnd配套使用，HcclGroupStart在前，HcclGroupEnd在后。

## 调用示例

- 示例一：单进程多线程管理NPU

    ```c
    HcclComm hccl_comms[devCount];
    HcclGroupStart();
    for(int i = 0; i &lt; ndev; i++){
        //
            aclrtSetDevice(i);
            HcclCommInitRootInfo(devCount, &rootInfo, global_rank, &(hccl_comms[i]));
        }
    HcclGroupEnd();
    ```

- 示例二：合并多个集合通信操作

    ```c
    HcclGroupStart();
        HCCLCHECK(HcclReduceScatter(sendBuf, recvBuf, 1, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream));
        HCCLCHECK(HcclAllGather(recvBuf, sendBuf, 1, HCCL_DATA_TYPE_FP32, hcclComm, stream));
    HcclGroupEnd();
    ```

- 示例三：合并多个点对点通信

    ```c
    HcclGroupStart();
    for(int i = 0; i &lt; devCount; i++){
        HCCLCHECK(HcclSend(sendBuf[i], count, HCCL_DATA_TYPE_FP32, i, hcclComm, stream));
        }
    for(int i = 0; i &lt; devCount; i++){
        HCCLCHECK(HcclRecv(recvBuf[i], count, HCCL_DATA_TYPE_FP32, i, hcclComm, stream));
        }
    HcclGroupEnd();
    ```
