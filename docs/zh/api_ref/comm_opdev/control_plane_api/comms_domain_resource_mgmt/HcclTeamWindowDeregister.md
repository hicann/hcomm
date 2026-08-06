# HcclTeamWindowDeregister

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
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

注销一个内存窗口，从world team的window列表中移除该window记录并销毁Hcomm层window资源。

## 函数原型

```c
HcclResult HcclTeamWindowDeregister(HcommTeamHandle worldTeam, HcommWindowHandle window)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| worldTeam | 输入 | world team句柄，不可为NULL。 |
| window | 输入 | 待注销的窗口句柄，不可为NULL。可通过[HcclTeamWindowRegister](HcclTeamWindowRegister.md)创建。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 接口会取入参team对应的world team，从其window列表移除该window记录。

2. 不注销localMem的内存注册（由通信域析构兜底清理），不处理syncMem（team粒度，team销毁时释放）。

3. 若team未注册（可能已销毁），接口打印告警并返回HCCL_SUCCESS。

## 调用示例

```c
// 注销window
HcclResult ret = HcclTeamWindowDeregister(worldTeam, window);
if (ret != HCCL_SUCCESS) {
    printf("HcclTeamWindowDeregister failed, ret = %d\n", ret);
}
```
