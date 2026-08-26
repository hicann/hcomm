# HcommChannelGetStatus

<!-- md-trans-meta sourceCommit=8599a3a9125cf5f0c5888fc169802126f18fc5a4 translatedAt=2026-08-14T09:11:09.124Z pushedAt=2026-08-17T02:23:31.651Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Queries the link establishment status of communication channels and advances the link establishment state machine by one step. Each call performs one round of link establishment advancement for all channels passed in and returns the current status. The caller must call this API repeatedly until the channel status becomes ready or failed. Communication operations can be performed only after the channel is ready.


## Function Prototype

```c
HcommResult HcommChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList);
```

## Parameters

| Parameter Name | Input/Output | Description |
| --- | --- | --- |
| channelList | Input | Array of channel handles whose status is to be queried. Each element identifies a created communication channel.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md).<br>This parameter cannot be a null pointer. Each channel handle in the array must be a valid handle created by [HcommChannelCreate](HcommChannelCreate.md). |
| listNum | Input | Number of channels to be queried.<br>Unit: number. Value range: [1, 1048576].<br>This parameter must be greater than 0. |
| statusList | Output | Array of channel statuses, used to return the current status of each channel. It corresponds to channelList one by one.<br>This parameter cannot be a null pointer.<br>The array is allocated by the caller and must contain space for at least listNum elements.<br>The status values are defined as follows:<br>0: Link establishment is complete and the channel is ready.<br>1: Link establishment is in progress and this API needs to be called to advance link establishment.<br>2: Link establishment failed.<br>3: Link establishment timed out. |

## Return Value

HcommResult: This API returns 0 upon success, and other values upon failure.

## Constraints

- The length of the channelList array must be consistent with the listNum parameter.
- statusList\[i\] corresponds to channelList\[i\] one by one, indicating the status of the i-th channel.
- The same ChannelHandle cannot be accessed concurrently in multiple threads. That is, the link establishment status query, communication, and destruction operations of the same channel must be executed serially.

## Example

```c
uint32_t channelNum = 50;
std::vector<ChannelHandle> channels(channelNum);
// Refer to HcommChannelCreate for channel creation.
...

std::vector<int32_t> statuses(channelNum, 0);
bool hasFailed = false;
while (true) {
    HcommResult ret = HcommChannelGetStatus(channels.data(), channelNum, statuses.data());
    if (ret != HCCL_SUCCESS) {
        // API call failed.
        break;
    }

    bool allReady = true;
    for (uint32_t i = 0; i < channelNum; i++) {
        if (statuses[i] == 2 || statuses[i] == 3) {
            // Link establishment failed or timed out. Exit.
            hasFailed = true;
            allReady = false;
            break;
        }
        if (statuses[i] != 0) {
            allReady = false;
        }
    }
    if (allReady || hasFailed) {
        break;
    }
    // Link establishment is in progress. Sleep for a while and retry.
    usleep(1000);
}

if (hasFailed) {
    // Link establishment failed or timed out. Perform error handling.
    // ...
    return;
}

 // The channel is ready. Perform communication operations.
 // ...
```
