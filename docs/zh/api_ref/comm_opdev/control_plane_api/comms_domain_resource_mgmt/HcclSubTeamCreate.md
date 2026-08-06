# HcclSubTeamCreate

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

基于已存在的worldTeam创建一个subTeam，用于HCCL通信。subTeam的成员必须是worldTeam成员的子集，创建成功后返回team句柄。

## 函数原型

```c
HcclResult HcclSubTeamCreate(HcommTeamHandle worldTeam, const HcclTeamCreateDesc *desc, HcommTeamHandle *team)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| worldTeam | 输入 | 父world team句柄，不可为NULL，需通过[HcclWorldTeamCreate](HcclWorldTeamCreate.md)创建。 |
| desc | 输入 | sub team创建描述符，不可为NULL，且rankIds不可为NULL。rankIds须为world team的rankIds子集。HcclTeamCreateDesc类型的定义可参见[HcclTeamCreateDesc](../../datatype_definition/HcclTeamCreateDesc.md)，使用前需通过[HcclTeamCreateDescInit](HcclTeamCreateDescInit.md)初始化。 |
| team | 输出 | 创建成功的subTeam句柄。HcommTeamHandle类型的定义可参见[HcommTeamHandle](../../datatype_definition/HcommTeamHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. desc需先通过[HcclTeamCreateDescInit](HcclTeamCreateDescInit.md)初始化。

2. rankNum不可为0或1（不支持创建单rank的team）。

3. sub team的rankIds必须是world team的rankIds的子集，否则返回HCCL_E_PARA。

4. selfRankId必须存在于rankIds中，否则返回HCCL_E_NOT_FOUND。

5. netLayer只能为0、1或2，0表示默认选择。

6. signalCount和counterCount目前只支持传0，暂不支持配置。

7. barrierCount必须大于等于1。

8. sub team无子team、无window，需通过[HcclTeamDestroy](HcclTeamDestroy.md)销毁。

## 调用示例

```c
// 假设worldTeam已通过HcclWorldTeamCreate创建，world team的rankIds为{0, 1, 2, 3}

// 1. 初始化sub team描述符（取world team的子集）
HcclTeamCreateDesc subDesc;
HcclTeamCreateDescInit(&subDesc);
uint32_t subRankIds[2] = {1, 3};
subDesc.rankIds = subRankIds;
subDesc.rankNum = 2;
subDesc.selfRankId = 1;
subDesc.requirement.signalCount = 0;
subDesc.requirement.counterCount = 0;
subDesc.requirement.barrierCount = 1;

// 2. 创建sub team
HcommTeamHandle subTeam = nullptr;
HcclResult ret = HcclSubTeamCreate(worldTeam, &subDesc, &subTeam);
if (ret != HCCL_SUCCESS) {
    printf("HcclSubTeamCreate failed, ret = %d\n", ret);
    return ret;
}

// 3. 销毁sub team
HcclTeamDestroy(subTeam);
```
