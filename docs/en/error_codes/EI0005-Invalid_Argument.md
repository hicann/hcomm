# EI0005 Invalid_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: operator name, group name, parameter name, local rank id, remote rank id.

```text
The arguments for collective communication are inconsistent between ranks, operator %s, group %s, parameter %s, local rank %s, remote rank %s.
```

Error example:

```text
The arguments for collective communication are inconsistent between ranks, operator HcomAllReduce, group hccl_world_group, parameter count, local rank 2176, remote rank 4224.
```

## Solution

Please modify the parameter value as prompted in the error message.
