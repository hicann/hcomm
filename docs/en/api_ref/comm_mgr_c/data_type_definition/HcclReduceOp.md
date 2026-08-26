# HcclReduceOp

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T08:25:24.353Z pushedAt=2026-08-14T08:48:16.195Z -->

## Description

Defines the types of reduce operations in collective communication.

## Prototype

```c
typedef enum {
    HCCL_REDUCE_SUM = 0,    /* sum */
    HCCL_REDUCE_PROD = 1,   /* prod */
    HCCL_REDUCE_MAX = 2,    /* max */
    HCCL_REDUCE_MIN = 3,    /* min */
    HCCL_REDUCE_RESERVED = 255 /* reserved */
} HcclReduceOp;
```
