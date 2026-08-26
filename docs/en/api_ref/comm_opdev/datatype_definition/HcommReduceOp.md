# HcommReduceOp

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T10:51:10.388Z pushedAt=2026-08-19T00:46:45.007Z -->

## Description

Defines the reduction operation type.

## Prototype

```c
typedef enum {
    HCOMM_REDUCE_SUM = 0,    /* sum */
    HCOMM_REDUCE_PROD = 1,   /* prod */
    HCOMM_REDUCE_MAX = 2,    /* max */
    HCOMM_REDUCE_MIN = 3,    /* min */
    HCOMM_REDUCE_RESERVED = 255  /* reserved */
} HcommReduceOp;
```
