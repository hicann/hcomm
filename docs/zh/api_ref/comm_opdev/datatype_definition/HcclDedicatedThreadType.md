# HcclDedicatedThreadType

## 功能说明

专用通信线程使用类型枚举，用于[HcclDedicatedThreadAcquire](../control_plane_api/comms_domain_resource_mgmt/HcclDedicatedThreadAcquire.md)接口指定专用线程的使用场景。

## 定义原型

```c
typedef enum {
    HCCL_DED_THREAD_TYPE_INVALID = -1,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH = 0,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE = 1,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE = 2,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH = 3,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE = 4,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE = 5
} HcclDedicatedThreadType;
```

## 成员说明

| 成员 | 描述 |
| --- | --- |
| HCCL_DED_THREAD_TYPE_INVALID | 无效的专用线程类型。 |
| HCCL_DED_THREAD_TYPE_AICPU_LAUNCH | 单算子用于AICPU任务下发的专用线程。 |
| HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE | 图模式下用于AICPU任务下发的专用线程。 |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE | 预留，当前不支持。 |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH | 预留，当前不支持。 |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE | 预留，当前不支持。 |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE | 预留，当前不支持。 |

## 约束说明

当前仅HCCL_DED_THREAD_TYPE_AICPU_LAUNCH与HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE在[HcclDedicatedThreadAcquire](../control_plane_api/comms_domain_resource_mgmt/HcclDedicatedThreadAcquire.md)接口中支持，其余成员为预留值，传入将返回失败。

