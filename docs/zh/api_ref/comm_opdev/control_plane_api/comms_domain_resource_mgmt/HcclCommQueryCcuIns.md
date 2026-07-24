# HcclCommQueryCcuIns

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

查询指定HCCL通信域已绑定的CCU实例，返回实例句柄用于后续CCU Kernel的注册与下发。

当前一个通信域最多绑定一个CCU实例，因此查询成功时固定返回1个实例：insHandles[0]为已绑定的实例句柄，*insNum为1。

> [!NOTE]注意
> 查询结果仅供借用，不转移实例所有权。返回的CCU实例所有权归通信域，由通信域负责释放；调用者不可销毁该实例，否则会造成对同一实例的重复释放，破坏通信域的资源管理。

## 函数原型

```c
HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄，不可为nullptr。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| insHandles | 输出 | CCU实例句柄数组，不可为nullptr。查询成功后insHandles[0]返回已绑定的CCU实例句柄。<br>CcuInsHandle类型的定义如下：<br>typedef uint64_t CcuInsHandle; |
| insNum | 输出 | CCU实例数量，不可为nullptr。查询成功后返回值固定为1。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

- 当comm、insHandles或insNum为nullptr时，返回HCCL_E_PTR。
- 当通信域代际不支持CCU（早于Ascend 950PR/Ascend 950DT）时，返回HCCL_E_NOT_SUPPORT。
- 当通信域未绑定CCU实例时，返回HCCL_E_UNAVAIL。

## 约束说明

1. CCU特性仅在Ascend 950PR/Ascend 950DT及以上代际支持，在更早代际的通信域上调用返回HCCL_E_NOT_SUPPORT。
2. 需在通信域完成初始化并绑定CCU实例后调用，否则返回HCCL_E_UNAVAIL。
3. 返回的CCU实例句柄仅供借用，所有权仍归通信域所有，由通信域负责释放。调用者不可销毁该实例，否则会造成对同一实例的重复释放，破坏通信域的资源管理。

## 调用示例

```c
// comm 需通过 HcclCommInitClusterInfo 等接口完成初始化，此处仅为示例占位
HcclComm comm = /* 已初始化的通信域句柄 */;
CcuInsHandle insHandle = 0;
uint32_t insNum = 0;

// 查询通信域已绑定的CCU实例
HcclResult ret = HcclCommQueryCcuIns(comm, &insHandle, &insNum);
// 当前实现固定返回1个CCU实例，insNum != 1视为失败
if (ret != HCCL_SUCCESS || insNum != 1) {
    // 错误处理：接口失败返回原始错误码，接口成功但实例数异常返回内部错误码
    return (ret != HCCL_SUCCESS) ? ret : HCCL_E_INTERNAL;
}

// 使用insHandle注册并下发CCU Kernel
CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
// ...
```
