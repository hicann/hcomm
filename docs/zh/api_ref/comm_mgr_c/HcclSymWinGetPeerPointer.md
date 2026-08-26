# HcclSymWinGetPeerPointer

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

根据对称内存窗口资源句柄和偏移，获取指定rank ID的对称内存窗口中该偏移对应的地址指针。

<!-- npu="A3" id6 -->
针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，本接口支持HCCS链路通信场景，返回对应的有效地址指针；若窗口为URMA模式，本接口将返回错误并将*ptr置为nullptr，此时应改用[HcclSymWinGetRemoteAddr](HcclSymWinGetRemoteAddr.md)获取地址。
<!-- end id6 -->

## 函数原型

```c
HcclResult HcclSymWinGetPeerPointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| winHandle | 输入 | 对称内存窗口资源句柄。<br>HcclCommSymWindow类型的定义可参见[HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md)。 |
| offset | 输入 | 使用[HcclCommSymWinGet](HcclCommSymWinGet.md)获取到的偏移量。 |
| peerRank | 输入 | rank ID，取值范围：[0, rankSize)。 |
| ptr | 输出 | 指向“对称内存窗口中对应地址”的指针。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

<!-- npu="A3" id7 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，仅支持HCCS链路通信场景。
<!-- end id7 -->
- 该接口仅支持通信算子展开模式为AI CPU的场景。
- 该接口仅支持在Device侧调用。

## 调用示例

HOST侧注册完对称内存window以后，将window作为AI CPU kernel的参数传递到AI CPU kernel，该函数需要编译到Device AI CPU侧执行，以下是伪代码描述：

```c
AicpuKernelFunc(param)：
// 从param中获取对称window
HcclCommSymWindow temp_win = param.win;
void *src_ptr;
void *dest_ptr;
int srcRankId = 0;
int destRankId = 1;
// 使用win + offset + peerRank获取到peerRank对应的地址
HcclSymWinGetPeerPointer(temp_win, 0, srcRankId, &src_ptr);
HcclSymWinGetPeerPointer(temp_win, 0, destRankId, &dest_ptr);
// 获取出来的地址可以使用数据面local copy直接读写，thread和size需调用者准备好
HcommLocalCopyOnThread(thread, dest_ptr, src_ptr, size);
```
