# HcommEndpointCreate

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

创建通信设备Endpoint。

## 函数原型

```c
HcclResult HcommEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| endpoint | 输入 | Endpoint初始化配置信息。<br>EndpointDesc类型的定义请参见[EndpointDesc](../../datatype_definition/EndPointDesc.md)。 |
| endpointHandle | 输出 | 返回的Endpoint句柄。<br>EndpointHandle类型的定义请参见[EndpointHandle](../../datatype_definition/EndpointHandle.md)。 |

## 返回值

HcommResult：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

支持的通信协议包括：RoCE、UBC_TP、UBC_CTP、UBoE。

## 调用示例

```c
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_UBC_TP,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 100}}
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_DEVICE,
        .device = {
            .devPhyId = 0,
            .superDevId = 0,
            .serverIdx = 0,
            .superPodIdx = 0
        }
    },
    .raws = {0}
};
EndpointHandle endpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&endpointDesc, &endpointHandle);
```
