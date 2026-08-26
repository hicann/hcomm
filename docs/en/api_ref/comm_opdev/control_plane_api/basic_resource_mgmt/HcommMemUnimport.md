# HcommMemUnimport

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:15:03.396Z pushedAt=2026-08-17T03:05:34.907Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Disables memory import.

## Function Prototype

```c
HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Endpoint handle.<br>For the definition of the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| memDesc | Input | Description information pointer. |
| descLen | Input | Length of the description information. |

## Return Value

HcommResult: This API returns 0 on success and a non-zero value on failure.

## Constraints

None

## Example

```c
// Operations on the export side
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 100}}
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_DEVICE,
        .device = {
            .devPhyId = 0,
            .superDevId = 0,
            .serverIdx = 0,
            .superPodIdx = 0
        }
    },
    .raws = {0}
};
EndpointHandle endpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&endpointDesc, &endpointHandle);
const char *memTag = "HcclBuffer";
CommMem mem = {
    .type = COMM_MEM_TYPE_DEVICE,
    .addr = reinterpret_cast<void*>(0x1111),
    .size = 100
};
HcommMemHandle memHandle;
result = HcommMemReg(endpointHandle, memTag, &mem, &memHandle);

uint32_t memDescLen = 0;
void* memDesc = nullptr;
result = HcommMemExport(endpointHandle, memHandle, &memDesc, &memDescLen);

// Operations on the import side
CommMem outMem;
result = HcommMemImport(endpointHandle, memDesc, memDescLen, &outMem);

result = HcommMemUnimport(endpointHandle, memDesc, memDescLen);
```
