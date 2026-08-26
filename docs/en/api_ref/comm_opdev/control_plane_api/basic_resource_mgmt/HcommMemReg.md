# HcommMemReg

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:14:36.487Z pushedAt=2026-08-17T03:03:46.492Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Registers memory to a specified endpoint.

## Function Prototype

```c
HcommResult HcommMemReg(EndpointHandle endpointHandle, const char *memTag, const CommMem *mem, HcommMemHandle *memHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| memTag | Input | String identifier of the memory. |
| mem | Input | Memory description, including the physical location type of the memory, memory address, and memory region size in bytes.<br>For details about the CommMem type, see [CommMem](../../datatype_definition/CommMem.md). |
| memHandle | Output | Handle of the registered memory. |

## Return Value

HcommResult: The API returns 0 on success and a non-zero value on failure.

## Constraints

None

## Example

```c
struct in_addr ipAddr;
inet_pton(AF_INET, "192.168.1.100", &ipAddr);
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = ipAddr
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
```
