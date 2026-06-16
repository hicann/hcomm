# HcclConfig

## 功能说明

定义集合通信相关配置。

## 定义原型

```c
typedef enum {
    HCCL_DETERMINISTIC = 0, /* 0: non-deterministic, 1: deterministic */
    HCCL_CONFIG_RESERVED
} HcclConfig;
```

## 参数说明

- HCCL_DETERMINISTIC：是否开启确定性计算。

  - 0：不开启确定性计算。
  - 1：开启确定性计算。

    此参数仅在如下型号中支持配置：

  - Atlas A3 训练系列产品/Atlas A3 推理系列产品
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品
  - Atlas 推理系列产品
  - Atlas 训练系列产品

- HCCL_CONFIG_RESERVED：预留参数。
