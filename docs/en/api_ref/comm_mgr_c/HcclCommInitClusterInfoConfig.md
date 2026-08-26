# HcclCommInitClusterInfoConfig

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T08:34:21.348Z pushedAt=2026-08-14T09:42:14.893Z -->

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

Initializes HCCL based on the rank table and creates an HCCL communicator with specific configurations.

## Function Prototype

```c
HcclResult HcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank, HcclCommConfig *config, HcclComm *comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| clusterInfo | Input | Path of the rank table file (including the file name). As a string, the maximum length is 4096 bytes, including the terminator. |
| rank | Input | ID of the current rank.<br>Note that the value of this parameter must be consistent with the value of the corresponding "rank_id" field in the rank table. |
| config | Input | Communicator configuration items, including the buffer size, deterministic computation switch, communicator name, and communication operator expansion mode. The configuration parameters must be within the valid value range. For details about the meanings and priorities of the parameters in HcclCommConfig, see [HcclCommConfig](./data_type_definition/HcclCommConfig.md).<br>Note: The config to be passed in must be initialized by calling [HcclCommConfigInit](HcclCommConfigInit.md) first. |
| comm | Output | Returns the initialized communicator to the caller as a pointer.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

Repeated initialization of the same communicator is not supported.

## Example

```c
// Initialize device resources.
aclInit(NULL);
// Path of the rank table configuration file
const char *rankTableFile = "/path/to/rank_table.json";
// Device used for collective communication operations
uint32_t rankSize = 8;
uint32_t devId = 0;
aclrtSetDevice(devId);
// Create and initialize the communicator configuration.
HcclCommConfig config;
HcclCommConfigInit(&config);
// Modify the communicator configuration as needed
config.hcclBufferSize = 50;  // Buffer size for shared data, in MB. The value must be >= 1. The default value is 200.
std::strcpy(config.hcclCommName, "comm_1");
// Initialize the communicator.
HcclComm hcclComm;
// In this example, devId is used as the rank ID of the current rank.
HcclCommInitClusterInfoConfig(rankTableFile, devId, &config, &hcclComm);
// Destroy the communicator.
HcclCommDestroy(hcclComm);
// Deinitialize the device resources.
aclFinalize();
```
