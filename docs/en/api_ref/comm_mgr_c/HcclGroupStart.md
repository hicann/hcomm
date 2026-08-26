# HcclGroupStart

<!-- md-trans-meta sourceCommit=1ed0535ba46025e54fb68f42726e7319b84ac5ca translatedAt=2026-08-14T08:52:11.141Z pushedAt=2026-08-15T07:24:01.875Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Starts a group call.

Multiple functions called between HcclGroupStart and HcclGroupEnd are executed as a whole. A group call supports the following three scenarios:

- Managing NPUs in a single process with multiple threads: supports calling the communicator management APIs [HcclCommInitClusterInfo](HcclCommInitClusterInfo.md), [HcclCommInitClusterInfoConfig](HcclCommInitClusterInfoConfig.md), [HcclCommInitRootInfo](HcclCommInitRootInfo.md), [HcclCommInitRootInfoConfig](HcclCommInitRootInfoConfig.md), and [HcclCommDestroy](HcclCommDestroy.md).
- Merging multiple collective communication operations (not supported on Ascend 950PR/Ascend 950DT).
- Merging multiple point-to-point communication operations.

## Function Prototype

```c
HcclResult HcclGroupStart()
```

## Parameters

None

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The group call APIs can be used for communicator management only in a single-server environment.
- Within a single group call, APIs of the communicator management, collective communication, and point-to-point communication types cannot be mixed.
- When merging multiple point-to-point communication operations, the [HcclBatchSendRecv](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/api_ref/comm_op_interface/HcclBatchSendRecv.md) API is not supported.
- HcclGroupStart must be used together with HcclGroupEnd, with HcclGroupStart called first and HcclGroupEnd called last.

## Example

- Example 1: Managing NPUs in a single process with multiple threads

    ```c
    HcclComm hccl_comms[devCount];
    HcclGroupStart();
    for(int i = 0; i &lt; ndev; i++){
        //
            aclrtSetDevice(i);
            HcclCommInitRootInfo(devCount, &rootInfo, global_rank, &(hccl_comms[i]));
        }
    HcclGroupEnd();
    ```

- Example 2: Merging multiple collective communication operations

    ```c
    HcclGroupStart();
        HCCLCHECK(HcclReduceScatter(sendBuf, recvBuf, 1, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream));
        HCCLCHECK(HcclAllGather(recvBuf, sendBuf, 1, HCCL_DATA_TYPE_FP32, hcclComm, stream));
    HcclGroupEnd();
    ```

- Example 3: Merging multiple point-to-point communication operations

    ```c
    HcclGroupStart();
    for(int i = 0; i &lt; devCount; i++){
        HCCLCHECK(HcclSend(sendBuf[i], count, HCCL_DATA_TYPE_FP32, i, hcclComm, stream));
        }
    for(int i = 0; i &lt; devCount; i++){
        HCCLCHECK(HcclRecv(recvBuf[i], count, HCCL_DATA_TYPE_FP32, i, hcclComm, stream));
        }
    HcclGroupEnd();
    ```
