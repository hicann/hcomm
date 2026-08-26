# HcclGetHcclBuffer

<!-- md-trans-meta sourceCommit=66864ecea5b8ff5139f4241c95eddba61029462d translatedAt=2026-08-14T09:30:13.020Z pushedAt=2026-08-17T07:07:04.604Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

> [!NOTE] Note
> For Atlas A2 training products/Atlas A2 inference products, only the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster base unit, and Atlas 200T A2 Box16 heterogeneous subrack are supported.

## Description

Obtains the HCCL communication memory of the local rank in the communicator. On the first call, the memory is initialized and allocated on the device side. Subsequent calls reuse the allocated memory without re-initialization.

> [!NOTE] Caution
> The returned HCCL communication memory is managed internally by hcomm. The caller must not release it.

## Function Prototype

```c
HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| buffer | Output | HCCL communication memory address. |
| size | Output | HCCL communication memory size. The memory size is twice the value configured during communicator initialization or the value of the HCCL_BUFFSIZE environment variable, and is 400 MB by default. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm;
void *hcclBuffer = nullptr;
uint64_t hcclBufferSize = 1 * 1024;   // 1KB
HcclResult result = HcclGetHcclBuffer(comm, &hcclBuffer, &hcclBufferSize);
```
