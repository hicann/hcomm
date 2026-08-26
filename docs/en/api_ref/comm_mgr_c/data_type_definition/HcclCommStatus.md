# HcclCommStatus

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T08:21:03.727Z pushedAt=2026-08-14T08:22:12.472Z -->

## Description

Communicator status information.

## Prototype

```c
typedef enum {
    HCCL_COMM_STATUS_READY = 0,
    HCCL_COMM_STATUS_SUSPENDING = 1,
    HCCL_COMM_STATUS_INVALID = 254,
    HCCL_COMM_STATUS_RESERVED = 255
} HcclCommStatus;
```
