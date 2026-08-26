# HcommChannelDestroy

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:10:57.958Z pushedAt=2026-08-17T02:16:58.052Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

This API is a resource management API for destroying communication channels. It releases the communication channels created by [HcommChannelCreate](HcommChannelCreate.md) and all system resources occupied by them, including network connections, synchronization signals, and communication queues.

This API supports batch destruction and can release multiple channels in a single call, improving resource release efficiency.

## Function Prototype

```c
HcommResult HcommChannelDestroy(const ChannelHandle *channels, uint32_t channelNum);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channels | Input | Array of channel handles to be destroyed. Each element identifies a created communication channel.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md).<br>This parameter cannot be a null pointer. Each channel handle in the array must be a valid handle created by [HcommChannelCreate](HcommChannelCreate.md) (a handle that has not been destroyed). |
| channelNum | Input | Number of channels to be destroyed.<br>Unit: number. Value range: [1, 1048576].<br>This parameter must be greater than 0 and equal to the number of valid handles in the channels array. |

## Return Value

HcommResult: The API returns 0 on success and a non-zero value on failure.

## Constraints

- The length of the channels array must be consistent with the channelNum parameter.
- Channels created by different engines can be destroyed in batches.
- The destruction operation is executed one by one. If a failure occurs, the operation returns immediately. Destroyed channels are not rolled back, and the operation is non-atomic.

## Example

```c
EndpointHandle endpointHandle = nullptr;
 // ... Code for creating an endpoint (omitted)

 // Create multiple channels.
 const uint32_t CHANNEL_NUM = 4;
 HcommChannelDesc channelDescs[CHANNEL_NUM] = {0};
 ChannelHandle channels[CHANNEL_NUM] = {0};

 // Prepare channel descriptors and create channels.
 for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
     // ... Fill channelDescs[i].
 }

 HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU,
                                      channelDescs, CHANNEL_NUM, channels);
 if (ret != 0) {
     printf("Failed to create channels, ret = %d\n", ret);
     HcommEndpointDestroy(endpointHandle);
     return ret;
 }

 printf("%u channels created successfully\n", CHANNEL_NUM);

 // Use the channels for communication.
 // ...

 // Destroy all channels in batches.
 ret = HcommChannelDestroy(channels, CHANNEL_NUM);
 if (ret != 0) {
     printf("Failed to destroy channels, ret = %d\n", ret);
 } else {
     printf("All channels destroyed successfully\n");
 }

 // Destroy the endpoint.
 HcommEndpointDestroy(endpointHandle);
```
