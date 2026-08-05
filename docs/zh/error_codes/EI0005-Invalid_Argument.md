# EI0005 Invalid_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为算子名、通信域组、参数名、本端ID、对端ID：

```text
The arguments for collective communication are inconsistent between ranks, operator %s, group %s, parameter %s, local rank %s, remote rank %s.
```

报错示例如下：

```text
The arguments for collective communication are inconsistent between ranks, operator HcomAllReduce, group hccl_world_group, parameter count, local rank 2176, remote rank 4224.
```

## 解决方法

请根据报错提示调整参数值。
