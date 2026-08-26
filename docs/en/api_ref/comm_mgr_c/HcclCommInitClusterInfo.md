# HcclCommInitClusterInfo

<!-- md-trans-meta sourceCommit=4296112684f605f4a436db49d4fc4ee45c3b6646 translatedAt=2026-08-14T08:31:53.152Z pushedAt=2026-08-14T09:37:00.289Z -->

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

Initializes HCCL based on the rank table and creates an HCCL communicator.

The rank table file is a JSON-format file that configures the NPU resource information involved in collective communication. For details about the rank table file configuration, see the [cluster information configuration](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/cluster_info_config/README.md).

## Function Prototype

```c
HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| clusterInfo | Input | Path of the rank table file (including the file name). As a string, the maximum length is 4096 bytes, including the terminator. |
| rank | Input | ID of the current rank.<br>Note that the value of this parameter must be consistent with the value of the corresponding "rank_id" field in the rank table. |
| comm | Output | Returns the initialized communicator to the caller through a pointer.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

Repeated initialization reports an error.

## Example

```c
// Initialize the device resources.
aclInit(NULL);
// Path of the rank table configuration file.
char *rankTableFile = "/path/rank_table.json";
// Device ID used for the collective communication operation.
uint32_t devId = 0;
aclrtSetDevice(devId);
// Create a communicator.
HcclComm hcclComm;
// In this example, devId is used as the rank ID of the current rank.
HcclCommInitClusterInfo(rankTableFile, devId, &hcclComm);
// Destroy the communicator.
HcclCommDestroy(hcclComm);
// Deinitialize the device resources.
aclFinalize();
```
