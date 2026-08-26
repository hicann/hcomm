# HcclCommStateCallback

<!-- md-trans-meta sourceCommit=be5a20837ba5b45ac8b4d47a01300793e41db317 translatedAt=2026-08-14T08:21:08.833Z pushedAt=2026-08-14T08:26:17.609Z -->

## Description

Defines the callback function type to be called at different phases of a communicator.

## Prototype

```c
typedef HcclResult (*HcclCommStateCallback)(HcclComm comm, HcclCommStatePhase state, void *args)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. |
| state | Input | Different phases of the communicator. For the definition of the HcclCommStatePhase type, see [HcclCommStatePhase](./HcclCommStatePhase.md). |
| args | Input | User context pointer passed to the callback function. |
