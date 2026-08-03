# 简介

本节包含CCU Kernel Host侧生命周期管理接口及内存Token查询接口。

通过这些接口，用户可以完成CCU实例的创建与销毁、Kernel的注册与翻译、Kernel的启动执行，以及进程虚拟地址到CCU访问Token的转换。

## 前置条件

本节接口需要传入 `CcuInsHandle`（CCU 实例句柄），获取步骤如下：

1. **创建 HcclComm 通信域**：参见 [HcclCommInitClusterInfo](../../../comm_mgr_c/HcclCommInitClusterInfo.md)
   或 [HcclCommInitRootInfo](../../../comm_mgr_c/HcclCommInitRootInfo.md)。

2. **从通信域查询 CCU 实例句柄**：调用 `HcclCommQueryCcuIns`（声明于 `include/hccl/hccl_ccu_res.h`）：

    ```c
    extern HcclResult HcclCommQueryCcuIns(HcclComm comm,
        CcuInsHandle *insHandles, uint32_t *insNum);
    ```

   典型用法：

    ```c
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryCcuIns(comm, &insHandle, &insNum);
    // 当前实现固定返回 1 个 CCU 实例，insNum != 1 视为失败
    if (ret != HCCL_SUCCESS || insNum != 1) {
        // 错误处理
    }
    ```

> 该接口属于Hccl层（不在`Hcomm*`/`Ccu*`系列内），暂未提供独立API参考页面，
> 完整签名以头文件`include/hccl/hccl_ccu_res.h`为准。

## 接口调用顺序

接口调用的标准顺序如下：

1. [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)：开始一轮Kernel注册。
2. [HcommCcuKernelRegister](HcommCcuKernelRegister.md)：注册Kernel函数，记录其操作序列。
3. [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md)：结束本轮注册，翻译为设备指令并下发。
4. [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md)：启动Kernel执行（可重复调用）。

旁路接口：

- [HcommCcuGetMemToken](HcommCcuGetMemToken.md)：将进程虚拟地址转换为CCU可用的内存Token，与主流程无依赖关系，可在任意时机调用。

## 参见

- [CCU 快速上手（含 AllGather 完整流程）](../../../../comm_op_dev_guide/ccu_quick_start.md)
- [CCU 通信算子开发指南（分步详解）](../../../../comm_op_dev_guide/ccu_comm_op_dev/README.md)
