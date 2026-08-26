# HcclCommStatePhase

<!-- md-trans-meta sourceCommit=be5a20837ba5b45ac8b4d47a01300793e41db317 translatedAt=2026-08-14T08:21:07.301Z pushedAt=2026-08-14T08:24:37.579Z -->

## Description

Different phases of a communicator.

## Prototype

```c
typedef enum {
    HCCL_COMM_STATE_PHASE_INVALID = -1,
    HCCL_COMM_STATE_PHASE_DESTROY_PRE = 0,   /* Before calling HcclCommDestroy to destroy the communicator */
    HCCL_COMM_STATE_PHASE_DESTROY_POST = 1,  /* After calling HcclCommDestroy to destroy the communicator */
    HCCL_COMM_STATE_PHASE_RESUME_PRE = 2,    /* Before calling HcclCommResume to resume communicator resources through step fast recovery */
    HCCL_COMM_STATE_PHASE_RESUME_POST = 3    /* After calling HcclCommResume to resume communicator resources through step fast recovery */
} HcclCommStatePhase;
```
