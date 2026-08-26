# HcclConfigValue

<!-- md-trans-meta sourceCommit=937bf528154f21d3efd5a05add93790f126d28cc translatedAt=2026-08-14T08:22:30.026Z pushedAt=2026-08-14T08:33:42.760Z -->

## Description

Defines the value of a configurable parameter in [HcclConfig](HcclConfig.md).

## Prototype

```c
union HcclConfigValue {
    int32_t value;
};
```

**value** is the value of the "HCCL_DETERMINISTIC" parameter in [HcclConfig](HcclConfig.md).
