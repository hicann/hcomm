# HcclWorldTeamCreate

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

基于已初始化的通信域创建一个worldTeam，用于HCCL通信。worldTeam是subTeam、window、channel的父资源，创建成功后返回team句柄。

## 函数原型

```c
HcclResult HcclWorldTeamCreate(HcclComm comm, const HcclTeamCreateDesc *desc, HcommTeamHandle *worldTeam)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 已初始化的通信域句柄，不可为NULL。HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| desc | 输入 | world team创建描述符，不可为NULL，且rankIds不可为NULL。HcclTeamCreateDesc类型的定义可参见[HcclTeamCreateDesc](../../datatype_definition/HcclTeamCreateDesc.md)，使用前需通过[HcclTeamCreateDescInit](HcclTeamCreateDescInit.md)初始化。 |
| worldTeam | 输出 | 创建成功的worldTeam句柄。HcommTeamHandle类型的定义可参见[HcommTeamHandle](../../datatype_definition/HcommTeamHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. desc需先通过[HcclTeamCreateDescInit](HcclTeamCreateDescInit.md)初始化。

2. rankNum不可为0或1（不支持创建单rank的team），且不可大于通信域的rankSize。

3. selfRankId必须存在于rankIds中，否则返回HCCL_E_NOT_FOUND。

4. netLayer只能为0、1或2，0表示默认选择。

5. signalCount和counterCount目前只支持传0，暂不支持配置。

6. barrierCount必须大于等于1。

7. worldTeam是subTeam、window、channel的父资源，需通过[HcclTeamDestroy](HcclTeamDestroy.md)销毁。

## 调用示例

```c
// 1. 初始化描述符
HcclTeamCreateDesc desc;
HcclTeamCreateDescInit(&desc);
uint32_t rankIds[2] = {0, 1};
desc.rankIds = rankIds;
desc.rankNum = 2;
desc.selfRankId = 1;
desc.requirement.signalCount = 0;
desc.requirement.counterCount = 0;
desc.requirement.barrierCount = 1;

// 2. 创建world team
HcommTeamHandle worldTeam = nullptr;
HcclResult ret = HcclWorldTeamCreate(comm, &desc, &worldTeam);
if (ret != HCCL_SUCCESS) {
    printf("HcclWorldTeamCreate failed, ret = %d\n", ret);
    return ret;
}

// 3. 后续可注册window、创建channel等操作
// ...

// 4. 销毁team
HcclTeamDestroy(worldTeam);
```
