# HcclCommConfigInit

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-14T08:28:33.083Z pushedAt=2026-08-14T09:12:22.808Z -->

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

Initializes the communicator configuration.

## Function Prototype

```c
static inline void HcclCommConfigInit(HcclCommConfig *config)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Output | Communicator configuration to be initialized.<br>For the definition of the HcclCommConfig type, see [HcclCommConfig](./data_type_definition/HcclCommConfig.md). |

## Return Value

None

## Constraints

None

## Example

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// Generate the rank identifier information of the root node.
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);

// Create and initialize the communicator configuration.
HcclCommConfig config;
HcclCommConfigInit(&config);
// Modify the communicator configuration as needed
config.hcclBufferSize = 1024;  // Buffer size for shared data, in MB. The value must be >= 1. The default value is 200
config.hcclDeterministic = 1;  // Enable deterministic computation for reduction communication operators. The default value is 0, indicating that deterministic computation is disabled
std::strcpy(config.hcclCommName, "comm_1");
// Initialize the collective communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, deviceId, &config, &hcclComm));

// Destroy the communicator.
HcclCommDestroy(hcclComm);
```
