# CcuResult

<!-- md-trans-meta sourceCommit=cf1ca0fcca3200769daa03ac3cb18897f92dd1ab translatedAt=2026-08-14T10:45:22.247Z pushedAt=2026-08-18T11:47:31.832Z -->

## Description

Return value type of CCU APIs, used to indicate the execution result of an operation. All CCU control-plane and data-plane APIs return the operation status using this type.

## Prototype

```c
typedef enum {
    CCU_SUCCESS = 0,
    CCU_E_PARA = 1,
    CCU_E_PTR = 2,
    CCU_E_INTERNAL = 4,
    CCU_E_NOT_SUPPORT = 5,
    CCU_E_NOT_FOUND = 6,
    CCU_E_UNAVAIL = 7,
    CCU_E_RUNTIME = 15,
    CCU_E_DRV_START = 4096,
    CCU_E_DRV_INIT_FAILED = 4097,
    CCU_E_DRV_BUSY = 4098,
    CCU_E_DRV_END = 4224,
    CCU_E_RESERVED = 9216
} CcuResult;
```

## Enum Value Description

| Enum Value | Value | Description |
| --- | --- | --- |
| `CCU_SUCCESS` | 0 | Operation succeeded. |
| `CCU_E_PARA` | 1 | Invalid parameter. |
| `CCU_E_PTR` | 2 | Null pointer error. |
| `CCU_E_INTERNAL` | 4 | Internal error. |
| `CCU_E_NOT_SUPPORT` | 5 | Unsupported feature. |
| `CCU_E_NOT_FOUND` | 6 | Specified resource not found. |
| `CCU_E_UNAVAIL` | 7 | Resource unavailable. |
| `CCU_E_RUNTIME` | 15 | Runtime error. |
| `CCU_E_DRV_START` | 4096 | Start marker of the driver-layer error code range (not returned as an actual error code). |
| `CCU_E_DRV_INIT_FAILED` | 4097 | Driver initialization failed. |
| `CCU_E_DRV_BUSY` | 4098 | Driver busy. |
| `CCU_E_DRV_END` | 4224 | End marker of the driver-layer error code range (not returned as an actual error code). |
| `CCU_E_RESERVED` | 9216 | Start marker of the reserved error code range (not returned as an actual error code; reserved for future extension). |
