# HcommAcquireComm

<!-- md-trans-meta sourceCommit=5d2b6a1c1d3ecb2aedf23e593d18044c239c2ddc translatedAt=2026-08-14T10:40:53.256Z pushedAt=2026-08-18T11:34:59.212Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the corresponding communicator based on the passed commId and locks the communicator to prevent it from being obtained repeatedly.

## Function Prototype

```c
int32_t HcommAcquireComm(const char* commId)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| commId | Input | Communicator ID. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

1. This API can be called only on the device side in AI CPU mode.
2. HcommAcquireComm and HcommReleaseComm correspond to the lock and unlock actions respectively, and must be called in pairs. The API internally intercepts repeated lock scenarios to prevent the same communicator from being occupied by multiple threads simultaneously.

## Example

This function must be compiled for use on the device side:

```c
// Kernel function executed on the AI CPU
extern "C" unsigned int HcclAicpuKernel(const char* commId)
{
    // Lock the communicator to prevent it from being used concurrently.
    if (HcommAcquireComm(commId) != HCCL_SUCCESS) {
        return 1;
    }

    // Execute task orchestration.
    // ...

    // Release the communicator.
    if (HcommReleaseComm(commId) != HCCL_SUCCESS) {
        return 1;
    }
    return 0;
}
```
