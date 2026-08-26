# HcclConfig

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T08:22:17.119Z pushedAt=2026-08-14T08:31:47.669Z -->

## Description

Defines the configuration related to collective communication.

## Prototype

```c
typedef enum {
    HCCL_DETERMINISTIC = 0, /* 0: non-deterministic, 1: deterministic */
    HCCL_CONFIG_RESERVED
} HcclConfig;
```

## Parameters

- HCCL_DETERMINISTIC: Whether to enable deterministic computation.

  - 0: Disable deterministic computation.
  - 1: Enable deterministic computation.

  This parameter can be configured only on the following models:

  - Atlas A3 training products/Atlas A3 inference products
  - Atlas A2 training products/Atlas A2 inference products
  - Atlas inference products
  - Atlas training products

- HCCL_CONFIG_RESERVED: Reserved parameter.
