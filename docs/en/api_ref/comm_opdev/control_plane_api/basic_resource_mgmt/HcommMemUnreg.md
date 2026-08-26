# HcommMemUnreg

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:16:35.805Z pushedAt=2026-08-17T03:06:57.887Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Deregisters memory from an endpoint.

## Function Prototype

```c
HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| memHandle | Input | Handle of the registered memory. |

## Return Value

HcommResult: This API returns 0 on success and a non-zero value on failure.

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
result = HcommMemUnreg(endpointHandle, memHandle);
```
