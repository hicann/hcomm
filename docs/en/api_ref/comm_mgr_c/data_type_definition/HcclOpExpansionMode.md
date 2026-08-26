# HcclOpExpansionMode

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T08:25:02.296Z pushedAt=2026-08-14T08:47:51.840Z -->

## Description

Defines the expansion mode of a communication operator.

## Prototype

```c
typedef HcclOpExpansionMode HcclConfigTypeOpExpansionMode;

typedef enum {
    HCCL_OP_EXPANSION_MODE_INVALID  = -1,  /* Invalid mode, uninitialized or reserved. */
    HCCL_OP_EXPANSION_MODE_AI_CPU   = 0,   /* Expanded on the AI CPU on the device side. */
    HCCL_OP_EXPANSION_MODE_AIV      = 1,   /* Expanded on the Vector Core (AIV) on the device side. */
    HCCL_OP_EXPANSION_MODE_HOST     = 2,   /* Expanded on the host-side CPU, and the device side automatically selects the scheduler based on the hardware model. */
    HCCL_OP_EXPANSION_MODE_HOST_TS  = 3,   /* Expanded on the host-side CPU, where the host dispatches tasks to the Device Task Scheduler, and the device side performs scheduling and execution. */
    HCCL_OP_EXPANSION_MODE_CCU_MS   = 4,   /* Expanded on the device-side Collective Communication Unit (CCU) using the Memory Slice (MS) mode. */
    HCCL_OP_EXPANSION_MODE_CCU_SCHED = 5,  /* Expanded on the device-side CCU using the scheduling mode (the CCU acts as the scheduler to dispatch tasks to the UB engine). */
    HCCL_OP_EXPANSION_AIV_ONLY      = 6,   /* Expanded only on the device-side Vector Core (AIV), without mode switching as the data size changes. */
} HcclOpExpansionMode;
```
