# HcommMemExport

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:14:09.752Z pushedAt=2026-08-17T02:52:44.492Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

After memory registration, exports the description of the specified memory for exchange.

## Function Prototype

```c
HcommResult HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Endpoint handle.<br>For the definition of the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| memHandle | Input | Handle of the registered memory. |
| memDesc | Output | Pointer to the returned description information. |
| memDescLen | Output | Length of the returned description information. |

## Return Value

HcommResult: The API returns 0 on success and a non-zero value on failure.

## Constraints

None

## Example

```c
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
```
