# HcommChannelFenceOnThread

<!-- md-trans-meta sourceCommit=cdd37e2a25803960d0d5c8bcee837044ac79fe73 translatedAt=2026-08-14T10:19:26.570Z pushedAt=2026-08-18T08:14:10.655Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Inserts a memory fence operation on a specified communication thread and communication channel to ensure that channel read/write operations before the fence are completed before the channel read/write operations after the fence.

## Function Prototype

```c
int32_t HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle. When called on the AI CPU side, this is the threads obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API. When called on the host CPU side, this parameter has no effect and can be set to 0.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channels obtained through the [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. For constraints on channel, see Constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

- For Ascend 950PR/Ascend 950DT, calling on both the AI CPU side and the host CPU side is supported.
- When called on the AI CPU side, the communication engine is AICPU_TS, and the communication protocols UBC_TP, UBC_CTP, and UBoE are supported.
- When called on the host CPU side, if [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) is called to allocate the input parameter `channel`, `engine = COMM_ENGINE_CPU` must be passed in, and `channelDesc.remoteEndpoint.protocol` must be `COMM_PROTOCOL_ROCE`, `COMM_PROTOCOL_UBC_TP`, or `COMM_PROTOCOL_UBC_CTP`.
- When called on the host CPU side, the `thread` parameter has no effect and can be set to 0.

## Example

### Collective Communication Example

```c
// Allocate communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS; // Used by Ascend 950PR/Ascend 950DT
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// Obtain the local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// Obtain the remote communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// Write the local memory content to the remote memory.
HcommWriteOnThread(thread, channel, remoteBuffer, localBuffer, len);
HcommChannelFenceOnThread(thread, channel);
HcommReadOnThread(thread, channel, localBuffer, remoteBuffer, len);
```

### Basic Communication Example

Take the host RoCE H2D scenario as an example. The client is the write initiator and provides the host source memory; the server is the write target and provides the device destination memory.

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

// Allocate server communication channel resources.
HcommChannelDesc serverChannelDesc = {};
result = HcommChannelDescInit(&serverChannelDesc, 1);
serverChannelDesc.remoteEndpoint = clientEndpointDesc;
serverChannelDesc.notifyNum = 1;
serverChannelDesc.exchangeAllMems = true;
serverChannelDesc.role = HCOMM_SOCKET_ROLE_SERVER;
serverChannelDesc.roceAttr.queueNum = 1;
ChannelHandle serverChannel = 0;
result = HcommChannelCreate(serverEndpointHandle, COMM_ENGINE_CPU, &serverChannelDesc, 1, &serverChannel);
```

#### Client

```c
// Allocate the client endpoint resource.
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
HcommResult result = HcommEndpointCreate(&clientEndpointDesc, &clientEndpointHandle);

// Register the client memory.
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
clientChannelDesc.exchangeAllMems = true;
clientChannelDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
clientChannelDesc.roceAttr.queueNum = 1;
ChannelHandle clientChannel = 0;
result = HcommChannelCreate(clientEndpointHandle, COMM_ENGINE_CPU, &clientChannelDesc, 1, &clientChannel);

// Write the client memory to the server memory.
int32_t ret = HcommWriteNbiOnThread(0, clientChannel, importedServerDeviceMem.addr, clientHostMem.addr, dataLen);

// Wait for the submitted write operations on the communication channel to complete.
ret = HcommChannelFenceOnThread(0, clientChannel);
```
