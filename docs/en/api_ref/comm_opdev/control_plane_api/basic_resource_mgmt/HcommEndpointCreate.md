# HcommEndpointCreate

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:11:45.118Z pushedAt=2026-08-17T02:47:38.101Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Creates an endpoint for a communication device.

## Function Prototype

```c
HcommResult HcommEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpoint | Input | Endpoint initialization configuration information.<br>Mandatory members: protocol, commAddr, and loc.<br>For details about the EndpointDesc type, see [EndpointDesc](../../datatype_definition/EndpointDesc.md). |
| endpointHandle | Output | Returned Endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |

## Return Value

HcommResult: The API returns 0 on success and a non-zero value on failure.

## Constraints

 - When the endpoint is on the host side, the supported protocols are RoCE, UBC_TP, and UBC_CTP.
 - When the endpoint is on the device side, the supported protocols are RoCE, UBC_TP, UBC_CTP, UB_MEM, PCIe, UBoE, and HCCS.

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
```
