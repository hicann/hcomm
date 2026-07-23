# HcclGetHcclBufferCleared

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

获取通信域中本端rank的HCCL通信内存，且获取到的通信内存已做内存清零，首次调用时将在Device侧进行内存初始化分配，后续调用将复用已分配内存，不会重复初始化。

> [!NOTE]注意
> 返回的HCCL通信内存由hcomm内部管理，调用者严禁释放。

## 函数原型

```c
HcclResult HcclGetHcclBufferCleared(HcclComm comm, void **buffer, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| buffer | 输出 | HCCL通信内存地址。 |
| size | 输出 | HCCL通信内存大小。内存大小为通信域初始化配置或环境变量HCCL_BUFFSIZE配置值的2倍，默认为400MB。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无。

## 调用示例

```c
HcclComm comm; // 已初始化
void *hcclBuffer = nullptr;
uint64_t hcclBufferSize = 1 * 1024;   // 1KB
HcclResult result = HcclGetHcclBufferCleared(comm, &hcclBuffer, &hcclBufferSize);
```
