# HcommReadNbiOnThread

<!-- md-trans-meta sourceCommit=6d41cf8dd0b097be1993a10deb9a75d9b04739d9 translatedAt=2026-08-14T10:24:21.769Z pushedAt=2026-08-18T09:08:44.855Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Reads data from the specified memory on the channel. It reads memory data of length `len` from `src` and writes it to the memory area of the same length pointed to by `dst`. The API caller is the node where `dst` resides. This API is a non-blocking API.

## Function Prototype

```c
int32_t HcommReadNbiOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle. It has no effect currently and can be set to 0. |
| channel | Input | Communication channel handle, which is the channel obtained through [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md). For constraints on the channel, see Constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address. In the basic communication scenario, the read initiator must use the local memory address registered through [HcommMemReg](../../../control_plane_api/basic_resource_mgmt/HcommMemReg.md). In the collective communication scenario, it is the local communication memory address obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address. In the basic communication scenario, it is the remote memory address imported through [HcommMemImport](../../../control_plane_api/basic_resource_mgmt/HcommMemImport.md). In the collective communication scenario, it is the remote communication memory address obtained through [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| len | Input | Data length (in bytes), which must be greater than 0. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

- For Ascend 950PR/Ascend 950DT, this API can be called only on the host CPU. When calling [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) to allocate the `channel` input parameter, set `engine = COMM_ENGINE_CPU`, and `channelDesc.remoteEndpoint.protocol` must be `COMM_PROTOCOL_ROCE`, `COMM_PROTOCOL_UBC_TP`, or `COMM_PROTOCOL_UBC_CTP`. Among them, the NBI read API (HcommReadNbiOnThread) and the NBI write API (HcommWriteNbiOnThread) support only the RoCE protocol (a DPU/1825 NIC must be configured), and do not support the UBC_CTP and UBC_TP protocols.
- When called on the host CPU, the `thread` parameter has no effect and can be set to 0.
- `[dst, dst + len)` must fall within the memory range registered or obtained on the local end, and `[src, src + len)` must fall within the memory range imported or obtained on the remote end.
- A successful return from this API only indicates that the read request has been submitted. To confirm that the read operation is complete, the caller should call [HcommChannelFenceOnThread](HcommChannelFenceOnThread.md) to wait for the submitted read operations on the channel to complete.

## Example

### Collective Communication Example

```c
// Omitted: Create the communicator handle comm.

// Allocate communication channel resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclResult result = HcclChannelDescInit(&channelDesc, channelNum);
channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
channelDesc.localEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
// Omitted: Fill in other information in channelDesc.
ChannelHandle channel;
result = HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// Obtain the local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
result = HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// Obtain the remote communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
result = HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// Read the content of the remote memory to the local memory.
int32_t ret = HcommReadNbiOnThread(0, channel, localBuffer, remoteBuffer, len);

// Wait for the submitted read operations on the communication channel to complete.
ret = HcommChannelFenceOnThread(0, channel);
```

### Basic Communication Example

Take the host RoCE D2H scenario as an example. The client is the read initiator and provides the host destination memory; the server is the read source and provides the device source memory.

#### Server

```c
// Allocate the server endpoint resource.
const EndpointDesc serverEndpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 101}}
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_HOST
    },
    .raws = {0}
};
EndpointHandle serverEndpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&serverEndpointDesc, &serverEndpointHandle);

// Register the server memory.
CommMem serverDeviceMem = {
    .type = COMM_MEM_TYPE_DEVICE,
    .addr = serverDeviceAddr,
    .size = dataLen
};
HcommMemHandle serverMemHandle = nullptr;
result = HcommMemReg(serverEndpointHandle, "server_device_mem", &serverDeviceMem, &serverMemHandle);

// Export the server memory description.
void *serverMemDesc = nullptr;
uint32_t serverMemDescLen = 0;
result = HcommMemExport(serverEndpointHandle, serverMemHandle, &serverMemDesc, &serverMemDescLen);

// Omitted:
// The server sends serverEndpointDesc, serverMemDesc, and serverMemDescLen to the client.
// The server obtains clientEndpointDesc.

// Allocate communication channel resources on the server.
HcommChannelDesc serverChannelDesc = {};
result = HcommChannelDescInit(&serverChannelDesc, 1);
serverChannelDesc.remoteEndpoint = clientEndpointDesc;
serverChannelDesc.notifyNum = 1;
serverChannelDesc.exchangeAllMems = true; // Automatically associate the local memHandle when the channel is created.
serverChannelDesc.role = HCOMM_SOCKET_ROLE_SERVER;
serverChannelDesc.roceAttr.queueNum = 1;
ChannelHandle serverChannel = 0;
result = HcommChannelCreate(serverEndpointHandle, COMM_ENGINE_CPU, &serverChannelDesc, 1, &serverChannel);
```

#### Client

```c
// Allocate the endpoint resource on the client.
const EndpointDesc clientEndpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 100}}
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_HOST
    },
    .raws = {0}
};
EndpointHandle clientEndpointHandle = nullptr;
result = HcommEndpointCreate(&clientEndpointDesc, &clientEndpointHandle);

// Register the memory on the client.
CommMem clientHostMem = {
    .type = COMM_MEM_TYPE_HOST,
    .addr = clientHostAddr,
    .size = dataLen
};
HcommMemHandle clientMemHandle = nullptr;
result = HcommMemReg(clientEndpointHandle, "client_host_mem", &clientHostMem, &clientMemHandle);

// Omitted:
// The client sends clientEndpointDesc to the server.
// The client obtains serverEndpointDesc, serverMemDesc, and serverMemDescLen.

// Import the server memory description.
CommMem importedServerDeviceMem = {};
result = HcommMemImport(clientEndpointHandle, serverMemDesc, serverMemDescLen, &importedServerDeviceMem);

// Allocate client communication channel resources.
HcommChannelDesc clientChannelDesc = {};
result = HcommChannelDescInit(&clientChannelDesc, 1);
clientChannelDesc.remoteEndpoint = serverEndpointDesc;
clientChannelDesc.notifyNum = 1;
clientChannelDesc.exchangeAllMems = true; // Automatically associate the local memHandle when the channel is created.
clientChannelDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
clientChannelDesc.roceAttr.queueNum = 1;
ChannelHandle clientChannel = 0;
result = HcommChannelCreate(clientEndpointHandle, COMM_ENGINE_CPU, &clientChannelDesc, 1, &clientChannel);

// Read the content of the server memory to the client memory.
int32_t ret = HcommReadNbiOnThread(0, clientChannel, clientHostMem.addr, importedServerDeviceMem.addr, dataLen);

// Wait for the read operations submitted on the communication channel to complete.
ret = HcommChannelFenceOnThread(0, clientChannel);
```
