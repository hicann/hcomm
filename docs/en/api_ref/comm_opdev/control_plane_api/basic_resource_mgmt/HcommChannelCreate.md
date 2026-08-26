# HcommChannelCreate

<!-- md-trans-meta sourceCommit=5d2b6a1c1d3ecb2aedf23e593d18044c239c2ddc translatedAt=2026-08-14T09:08:26.122Z pushedAt=2026-08-17T02:04:43.025Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

This API is a resource management API for creating communication channels. Based on the created network endpoints, it creates communication channels in batches according to the given channel description information, providing the data transmission infrastructure for point-to-point communication or collective communication.

After this API is executed, only the channel objects are created, and **link connections are not established immediately**. The caller needs to subsequently drive the link establishment state machine through the [HcommChannelGetStatus](HcommChannelGetStatus.md) API and perform communication operations after the channel state becomes ready.

## Function Prototype

```c
HcommResult HcommChannelCreate(EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Handle of the network device endpoint, which identifies a created local network device endpoint.<br>For the definition of the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). This handle must be successfully created through [HcommEndpointCreate](HcommEndpointCreate.md) and must not be destroyed. |
| engine | Input | Communication engine type, which specifies the execution location of the channel.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md).<br>Note: It must be a valid engine type. |
| channelDescs | Input | Array of channel descriptors, where each element describes the attribute information of a channel to be created.<br>For the definition of the HcommChannelDesc type, see [HcommChannelDesc](../../datatype_definition/HcommChannelDesc.md).<br>The number of array elements must be equal to channelNum, and each element must be correctly filled with the required fields. |
| channelNum | Input | Number of channels to be created.<br>Unit: number. Value range: [1, 1048576].<br>This parameter must be greater than 0. |
| channels | Output | Array of channel handles, used to return the list of handles of successfully created channels.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md).<br>An array allocated by the caller, which must contain space for at least channelNum elements. |

## Return Value

HcommResult: The API returns 0 on success and other values on failure.

## Constraints

- The length of the channelDescs array must be consistent with the channelNum parameter.
- The remoteEndpoint in HcommChannelDesc must be correctly filled with the remote endpoint information.
- When exchangeAllMems in HcommChannelDesc is false, memHandles and memHandleNum must be configured.
- When the current CommEngine is configured as CCU, only one memHandle can be exchanged.
- When the current CommEngine is configured as CCU, external configuration of NotifyNum is not supported, and the default is 8 CCU Notify.
- The communication protocols supported by each CommEngine are related to the chip model, as follows:

  For Ascend 950PR/Ascend 950DT, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_CPU
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_UBOE
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_AIV
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    - COMM_PROTOCOL_ROCE

  **Note:**
    - The Atlas 350 accelerator card does not support the COMM_ENGINE_CPU communication engine and its corresponding communication protocols.
    - The Atlas 350 accelerator card does not support the COMM_PROTOCOL_ROCE communication protocol of the COMM_ENGINE_AICPU_TS communication engine.

## Example

```c
EndpointHandle endpointHandle = nullptr;
 // ... Code for creating the endpoint (omitted)

 // Create multiple channels
 const uint32_t CHANNEL_NUM = 4;
 HcommChannelDesc channelDescs[CHANNEL_NUM] = {0};
 ChannelHandle channels[CHANNEL_NUM] = {0};

 // Prepare the channel descriptors and create the channels.
 for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
     // ... Fill channelDescs[i]
 }

 HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU,
                                      channelDescs, CHANNEL_NUM, channels);
 if (ret != 0) {
     printf("Failed to create channels, ret = %d\n", ret);
     HcommEndpointDestroy(endpointHandle);
     return ret;
 }

 printf("%u channels created successfully\n", CHANNEL_NUM);
```
