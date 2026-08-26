# HcclConfigType

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T08:22:24.212Z pushedAt=2026-08-14T08:32:29.066Z -->

## Description

Configures the expansion mode of communication operators.

## Prototype

```c
typedef enum {
    HCCL_CONFIG_TYPE_INVALID             = -1,   /* Invalid configuration type */
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE   = 0,    /* Operator expansion mode, corresponding to the hcclOpExpansionMode type */
} HcclConfigType;
```
