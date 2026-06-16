# HcclRankGraphGetEndpointDesc

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

获取拓扑实例的EndPoint描述列表。

## 函数原型

```c
HcclResult HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t *descNum, EndpointDesc *endpointDesc);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| layer | 输入 | 拓扑层级编号。 |
| topoInstId | 输入 | 拓扑实例ID。 |
| descNum | 输入/输出 | 要获取的通信设备描述数量。<br>作为输入时，需要等于[HcclRankGraphGetEndpointNum](HcclRankGraphGetEndpointNum.md)接口的输出“num”的值。 |
| endPointDesc | 输出 | EndPoint描述列表，需要调用方分配内存。<br>EndpointDesc类型的定义请参见[EndpointDesc](../../datatype_definition/EndPointDesc.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
HcclComm comm;
uint32_t layer = 0;
uint32_t topoInstId = 0;
uint32_t num = 0;
HcclRankGraphGetEndpointNum(comm, layer, topoInstId, &num);
uint32_t descNum = num;
HcclRankGraphGetEndpointDesc(comm, layer, topoInstId, &descNum, endPointDesc);
```
