# HcclTeamWindowRegister

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

为HCCL通信注册一个内存窗口。需要保证所有team上的rank都调用该接口，且只能给put接口使用，入参team仅能为world team。

HcclTeamWindowRegister生成的window必须调用[HcclTeamChannelsCreate](HcclTeamChannelsCreate.md)之后才能生效，如果只创建不调用则无法使用该window。

## 函数原型

```c
HcclResult HcclTeamWindowRegister(HcclComm comm, HcommTeamHandle worldTeam,
                                  const CommMem *localMem, HcommWindowHandle *window, uint32_t flag)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄，不可为NULL。HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| worldTeam | 输入 | world team句柄，不可为NULL，必须为world team（sub team不可用）。可通过[HcclWorldTeamCreate](HcclWorldTeamCreate.md)创建。 |
| localMem | 输入 | 用户本地内存描述。type必须为COMM_MEM_TYPE_DEVICE，addr不可为NULL，size不可为0。CommMem类型的定义可参见[CommMem](../../datatype_definition/CommMem.md)。 |
| window | 输出 | 注册成功的内存窗口句柄。HcommWindowHandle类型的定义可参见[HcommWindowHandle](../../datatype_definition/HcommWindowHandle.md)。 |
| flag | 输入 | 窗口标志，当前仅支持传0。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 入参worldTeam必须为world team，传sub team会返回HCCL_E_PARA。

2. 需要保证所有team上的rank都调用该接口，且各rank注册window的顺序必须一致，否则[HcclTeamChannelsCreate](HcclTeamChannelsCreate.md)交换远端内存时将因顺序不匹配导致window与远端内存错配。

3. flag当前仅支持传0，传其他值返回HCCL_E_PARA。

4. localMem.type必须为COMM_MEM_TYPE_DEVICE，addr不可为NULL，size不可为0。

5. window注册后不能直接使用，必须调用[HcclTeamChannelsCreate](HcclTeamChannelsCreate.md)之后才能生效。

6. 支持窗口复用：若world team已注册过window且本次localMem是某已注册window的localMem的子集，则复用该window句柄。

7. window归world team所有，可通过[HcclTeamWindowDeregister](HcclTeamWindowDeregister.md)注销。

## 调用示例

```c
// 准备本地内存
CommMem localMem;
localMem.type = COMM_MEM_TYPE_DEVICE;
localMem.addr = deviceMemPtr;  // 设备侧内存地址
localMem.size = memSize;

// 注册window（所有rank都需调用）
HcommWindowHandle window = nullptr;
HcclResult ret = HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, 0);
if (ret != HCCL_SUCCESS) {
    printf("HcclTeamWindowRegister failed, ret = %d\n", ret);
    return ret;
}

// 必须调用HcclTeamChannelsCreate后window才能生效
// HcclTeamChannelsCreate(comm, worldTeam, &channelsDesc);
```
