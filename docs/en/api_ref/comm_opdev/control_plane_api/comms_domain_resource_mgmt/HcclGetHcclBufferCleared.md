# HcclGetHcclBufferCleared

<!-- md-trans-meta sourceCommit=b7c872dbd6d19a623d6f4dfc69815e0c9e0826de translatedAt=2026-08-14T09:30:43.399Z pushedAt=2026-08-17T07:18:48.221Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the HCCL communication memory of the local rank in a communicator. The obtained communication memory has been cleared. On the first call, the memory is initialized and allocated on the device side. Subsequent calls reuse the allocated memory without re-initialization.

> [!NOTE] Note
> The returned HCCL communication memory is managed internally by hcomm. The caller must not release it.

## Function Prototype

```c
HcclResult HcclGetHcclBufferCleared(HcclComm comm, void **buffer, uint64_t *size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| buffer | Output | Address of the HCCL communication memory. |
| size | Output | Size of the HCCL communication memory. The memory size is twice the value configured during communicator initialization or the value configured by the HCCL_BUFFSIZE environment variable, and defaults to 400 MB. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm; // Initialized
void *hcclBuffer = nullptr;
uint64_t hcclBufferSize = 1 * 1024;   // 1KB
HcclResult result = HcclGetHcclBufferCleared(comm, &hcclBuffer, &hcclBufferSize);
```
