# HcommEndpointGetListenPort

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:14:05.479Z pushedAt=2026-08-17T02:50:43.470Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Obtains the listening port of the communication device endpoint.

## Function Prototype

```c
HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t *port)
```

## Parameters

| Parameter Name | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Endpoint handle.<br>For the definition of the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| port | Output | Listening port number returned. |

## Return Value

HcommResult: The API returns 0 on success and other values on failure.

## Constraints

- endpointHandle must be a valid handle created by [HcommEndpointCreate](HcommEndpointCreate.md).
- The port parameter cannot be a null pointer.

## Example

```c
struct in_addr ipAddr;
inet_pton(AF_INET, "192.168.1.100", &ipAddr);
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_UBC_TP,
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

uint32_t listenPort = 0;
result = HcommEndpointGetListenPort(endpointHandle, &listenPort);
```
