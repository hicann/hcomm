# HcommBatchTransferOnThread

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T10:17:35.560Z pushedAt=2026-08-18T07:45:27.261Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Asynchronously submits a group of transfer tasks on a specified thread and channel. Each transfer task is described by [HcommBatchTransferDesc](../../../datatype_definition/HcommBatchTransferDesc.md), and supports operation types such as one-sided write, one-sided read, write reduction, write with notification, and notification record/wait.

## Function Prototype

```c
int32_t HcommBatchTransferOnThread(ThreadHandle thread, ChannelHandle channel,
    const HcommBatchTransferDesc *transferDescs, uint32_t transferDescNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| transferDescs | Input | Array of batch transfer descriptors. Each element in the array describes one transfer task, and the tasks are submitted sequentially in array order. For the descriptor definition, see [HcommBatchTransferDesc](../../../datatype_definition/HcommBatchTransferDesc.md). |
| transferDescNum | Input | Number of descriptors, which must be greater than 0. |

## Return Value

int32_t: The API returns 0 on success and a non-zero value on failure.

## Constraints

For the following products, this API supports only the RoCE communication protocol, and transType supports only HCOMM_TRANSFER_TYPE_WRITE and HCOMM_TRANSFER_TYPE_READ. For other types, HCCL_E_NOT_SUPPORT is returned.

- Atlas A3 training products/Atlas A3 inference products
- Atlas A2 training products/Atlas A2 inference products

## Example

```c
// Allocate communication thread resources and channel resources (omitted).
// Obtain the local and remote communication memory information (omitted).

// Construct the batch transfer descriptors.
HcommBatchTransferDesc descs[2];

// Descriptor 1: one-sided write
descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
descs[0].transferInfo.write.len = 4096;
descs[0].transferInfo.write.dst = remoteBuffer;    // Remote address
descs[0].transferInfo.write.src = localBuffer;     // Local address

// Descriptor 2: one-sided write (with notification)
descs[1].transType = HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY;
descs[1].transferInfo.writeWithNotify.len = 8192;
descs[1].transferInfo.writeWithNotify.dst = remoteBuffer2;    // Remote address
descs[1].transferInfo.writeWithNotify.src = localBuffer2;     // Local address
descs[1].transferInfo.writeWithNotify.notifyIdx = 0;          // Notification index

// Submit the batch transfer.
HcommBatchTransferOnThread(thread, channel, descs, 2);
```
