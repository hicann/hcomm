# HcclCommInitAll

<!-- md-trans-meta sourceCommit=1ed0535ba46025e54fb68f42726e7319b84ac5ca translatedAt=2026-08-14T08:31:53.805Z pushedAt=2026-08-14T09:32:35.569Z -->

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
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

In a single-server communication scenario, a single process creates the communicators of multiple devices (one device corresponds to one thread). During communicator initialization, devices\[0\] serves as the root rank to automatically collect cluster information.

## Function Prototype

```c
HcclResult HcclCommInitAll(uint32_t ndev, int32_t*  devices, HcclComm* comms)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ndev | Input | Number of devices in the communicator. |
| devices | Input | List of devices in the communicator. The value is the logical ID of each device, which can be queried by running the npu-smi info -m command. HCCL creates the communicators in the order in which devices are set.<br>Note that the input device list cannot contain duplicate device IDs. |
| comms | Output | Array of generated communicator handles. Its size is ndev * sizeof(HcclComm).<br>For details about the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- This API is supported only in single-server communication scenarios, not in multi-server communication scenarios.
- When multiple threads call communication operation APIs (for example, HcclAllReduce), ensure that the time difference between the calls to communication operation APIs in different threads does not exceed the time specified by the environment variable [HCCL_CONNECT_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md) to avoid link establishment timeout.
- A single device cannot call multiple communication operation APIs at the same time.

## Example

```c
uint32_t rankSize = 2;
int32_t devices[rankSize] = {0, 1};
HcclComm comms[rankSize];
// Initialize the communicator.
HcclCommInitAll(rankSize, devices, comms);
// Destroy the communicator.
for (uint32_t i = 0; i &lt; rankSize; i++) {
    HcclCommDestroy(comms[i]);
}
```
