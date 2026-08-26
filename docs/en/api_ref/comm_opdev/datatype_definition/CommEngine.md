# CommEngine

<!-- md-trans-meta sourceCommit=d215080ac41ed2d19dfd6b3e03fcf0489d8e879c translatedAt=2026-08-14T10:46:27.737Z pushedAt=2026-08-18T11:50:31.116Z -->

## Description

Enumeration of communication engine types.

## Prototype

```c
typedef enum {
    COMM_ENGINE_RESERVED = -1,    //< Reserved communication engine
    COMM_ENGINE_CPU = 0,          //< HOST CPU engine
    COMM_ENGINE_CPU_TS = 1,       //< HOST CPU TS engine
    COMM_ENGINE_AICPU = 2,        //< AICPU engine
    COMM_ENGINE_AICPU_TS = 3,     //< AICPU TS engine
    COMM_ENGINE_AIV = 4,          //< AIV engine
    COMM_ENGINE_CCU = 5,          //< CCU engine
} CommEngine;
```

## Supported Products

For Ascend 950PR/Ascend 950DT, the supported communication engines are as follows:

  - COMM_ENGINE_CPU
  - COMM_ENGINE_AICPU_TS
  - COMM_ENGINE_AIV
  - COMM_ENGINE_CCU

For Atlas A3 training products/Atlas A3 inference products, the supported communication engines are as follows:

  - COMM_ENGINE_AICPU_TS

For Atlas A2 training products/Atlas A2 inference products, the supported communication engines are as follows:

  - COMM_ENGINE_CPU_TS
