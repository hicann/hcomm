# HcclTeamDestroy

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

销毁一个team（world team或sub team），释放其资源。销毁world team时会连带销毁其所有sub team与已注册的window，避免资源泄漏。

## 函数原型

```c
HcclResult HcclTeamDestroy(HcommTeamHandle team)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| team | 输入 | 待销毁的team句柄，不可为NULL。可通过[HcclWorldTeamCreate](HcclWorldTeamCreate.md)或[HcclSubTeamCreate](HcclSubTeamCreate.md)创建。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 销毁world team时会递归销毁其所有sub team，并销毁world team拥有的所有window（1:N），其中，sub team无子team、无window，因此无额外级联对象。

2. 接口会释放该team的syncMem本地内存并清理注册条目，但不注销window对应localMem的内存注册（由通信域析构兜底清理）。

3. team句柄销毁后不可再使用。

## 调用示例

```c
// 销毁world team（会连带销毁其所有sub team与window）
HcclResult ret = HcclTeamDestroy(worldTeam);
if (ret != HCCL_SUCCESS) {
    printf("HcclTeamDestroy failed, ret = %d\n", ret);
}
```
