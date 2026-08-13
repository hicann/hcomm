# HcclTeamChannelsCreate

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

为一个team创建channel（通道），用于HCCL通信。该接口是使已注册window生效的必要步骤：内部依次完成注册team的syncMem内存、获取world team及其所有window、对每个对端成员创建channel、绑定channel到team、收集远端内存并绑定到window与syncMem。

## 函数原型

```c
HcclResult HcclTeamChannelsCreate(HcclComm comm, HcommTeamHandle team,
                                  const HcclTeamCreateChannelsDesc *desc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 已初始化的通信域句柄，不可为NULL。HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| team | 输入 | team句柄（world team或sub team），不可为NULL，且须属于当前comm。可通过[HcclWorldTeamCreate](HcclWorldTeamCreate.md)或[HcclSubTeamCreate](HcclSubTeamCreate.md)创建。 |
| desc | 输入 | channel创建描述符，不可为NULL。HcclTeamCreateChannelsDesc类型的定义可参见[HcclTeamCreateChannelsDesc](../../datatype_definition/HcclTeamCreateChannelsDesc.md)，使用前需通过[HcclTeamCreateChannelsDescInit](HcclTeamCreateChannelsDescInit.md)初始化。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 本接口仅支持AIV引擎和URMA通信协议，暂不支持其他场景。

2. desc需先通过[HcclTeamCreateChannelsDescInit](HcclTeamCreateChannelsDescInit.md)初始化。

3. channelCnt在[HcclTeamCreateChannelsDesc](../../datatype_definition/HcclTeamCreateChannelsDesc.md)中指定，不可为0。

4. 若需使用window（put接口），必须先调用[HcclTeamWindowRegister](HcclTeamWindowRegister.md)注册window，再调用本接口，window才会生效。各rank注册window的顺序必须一致，否则接口内部交换远端内存时将因顺序不匹配导致window与远端内存错配。

5. 接口内部会对每个对端成员创建channelCnt个channel，并通过[HcclRankGraphGetLinks](../topo_info_query/HcclRankGraphGetLinks.md)获取链路信息。若rank graph中本rank到对端rank找不到link，返回HCCL_E_NOT_FOUND。

6. syncMem内存注册为team粒度，仅首次调用时注册一次。

## 调用示例

```c
// 1. 初始化channel创建描述符
HcclTeamCreateChannelsDesc channelsDesc;
HcclTeamCreateChannelsDescInit(&channelsDesc);
channelsDesc.engine = COMM_ENGINE_AICPU_TS;
channelsDesc.notifyNum = 8;
channelsDesc.protocol = COMM_PROTOCOL_UBOE;
channelsDesc.channelCnt = 1;

// 2. 创建channel（若已注册window，则window在此后生效）
HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &channelsDesc);
if (ret != HCCL_SUCCESS) {
    printf("HcclTeamChannelsCreate failed, ret = %d\n", ret);
    return ret;
}
```
