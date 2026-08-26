# HcclCommSymWinRegister

<!-- md-trans-meta sourceCommit=4296112684f605f4a436db49d4fc4ee45c3b6646 translatedAt=2026-08-14T08:41:26.038Z pushedAt=2026-08-15T03:45:13.324Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Registers service memory as a symmetric memory window, allowing HCCL to directly use this memory in supported collective communication operators.

Symmetric memory is a memory management model that allows parallel processing units (for example, each rank) to access each other's memory in a "globally visible" manner without explicit address exchange.

Currently, symmetric memory supports the following scenarios:

- HCCS scenario of Atlas A3 training products/Atlas A3 inference products: After allocating virtual memory and physical memory and completing the mapping, the user registers the virtual memory as symmetric memory. In this scenario, symmetric memory is implemented by reserving virtual addresses of the same size and same layout in advance.
- URMA scenario of Ascend 950PR/Ascend 950DT: The user registers the allocated device memory as a symmetric memory window. In this scenario, HcclCommSymWinRegister only completes the local symmetric memory window registration; cross-rank memory registration, memHandle exchange, and remote memory information update are completed when the relevant UB/URMA communication channel is created. When using collective communication APIs, this process is triggered internally by the collective communication operator. Before the remote memory information update is completed, do not call [HcclSymWinGetPeerPointer](HcclSymWinGetPeerPointer.md) to obtain the remote address.

The following figure shows the basic implementation model of symmetric memory in the HCCS scenario of Atlas A3 training products/Atlas A3 inference products.

![Symmetric memory implementation model](./figures/symmetric_memory.png)

- Allocate virtual memory for each rank. Assume that the virtual memory size corresponding to each rank is heap_size and the number of ranks in the communicator is rank_size. Then the total virtual memory size in the communicator is heap_size\*rank_size.
- The virtual address layout of each rank is the same.
- The physical addresses of different ranks are mapped to the virtual addresses at the corresponding positions of each rank, enabling access to the memory of other ranks.

The symmetric memory feature allows HCCL to directly operate on the memory passed in by the service without going through an intermediate buffer (HCCL buffer), thereby reducing memory copy overhead.

## Function Prototype

```c
HcclResult HcclCommSymWinRegister(HcclComm comm, void *addr, uint64_t size, HcclCommSymWindow *winHandle, uint32_t flag)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator.<br>In the HCCS scenario of Atlas A3 training products/Atlas A3 inference products, you are advised to use the largest communicator within the SuperPoD, that is, the communicator covering the maximum number of devices. When initializing the communicator, you can set the symmetric memory size reserved for each rank through the hcclSymWinMaxMemSizePerRank parameter of [HcclCommConfig](./data_type_definition/HcclCommConfig.md). If hcclSymWinMaxMemSizePerRank is not set, the default value 16 GB is used. The total virtual symmetric memory size reserved by the current communicator is: rankSize * HcclCommConfig.hcclSymWinMaxMemSizePerRank.<br>In the URMA scenario of Ascend 950PR/Ascend 950DT, you do not need to configure the reserved symmetric memory size through hcclSymWinMaxMemSizePerRank. |
| addr | Input | Start address of the symmetric memory window.<br>In the HCCS scenario of Atlas A3 training products/Atlas A3 inference products, this address is a reserved virtual memory address. The virtual memory needs to be reserved by calling the aclrtReserveMemAddress API.<br>In the URMA scenario of Ascend 950PR/Ascend 950DT, this address is an allocated device memory address. The memory must remain valid until [HcclCommSymWinDeregister](HcclCommSymWinDeregister.md) is called to deregister it. |
| size | Input | Size of the symmetric memory window.<br>In the HCCS scenario of Atlas A3 training products/Atlas A3 inference products, 0 < size <= HcclCommConfig.hcclSymWinMaxMemSizePerRank, and size cannot exceed the size of the "physical memory mapped to addr" (that is, the device physical memory allocated by calling the aclrtMallocPhysical API). Symmetric memory registration is aligned by the size of the physical memory, and the actual registered symmetric memory window size is equal to the size of the "physical memory mapped to addr".<br>In the URMA scenario of Ascend 950PR/Ascend 950DT, size must be greater than 0, and the size input by all ranks when calling this API must be consistent. |
| winHandle | Output | Pointer to the "symmetric memory window resource handle".<br>For the definition of the HcclCommSymWindow type, see [HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md). |
| flag | Input | Whether to enable symmetric memory. Currently, only 1 is supported. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- For Atlas A3 training products/Atlas A3 inference products:
  - Only the HCCS link communication scenario is supported.
  - Only symmetric networking is supported, that is, the scenario where each server has the same number of devices.
  - Only the scenario where AI servers within a SuperPoD use HCCS links for SDMA communication is supported. The scenario where RoCE is used for RDMA communication is not supported (that is, setting the environment variable HCCL_INTER_HCCS_DISABLE to "TRUE" is not supported; this environment variable is invalid in the single-server scenario).
  - Only collective communication operators AllGather, ReduceScatter, AllReduce, and AllToAll are supported.
  - The physical memory sizes mapped by the input addresses of all ranks must be the same (symmetric memory registration is aligned by physical memory size).

- For Ascend 950PR/Ascend 950DT:
  - Only the URMA scenario is supported.
  - Only the collective communication operator AllGather is supported.
  - The UB/URMA communication channels created inside the collective communication operator are required to complete symmetric memory resource registration and exchange. Users do not need to explicitly call HcclChannelAcquire.
  - The symmetric networking is not required.

- This API only supports the scenario where the communication operator expansion mode is AI CPU.
- Ensure that all ranks in the communicator call this registration API at the same time.
- When all ranks call this API, the input size parameter must be consistent.
- When using the symmetric memory feature, the input and output memory of operators must be registered as symmetric memory by calling this API.
- The memory registered by calling this API must be deregistered by calling [HcclCommSymWinDeregister](HcclCommSymWinDeregister.md).

## Example

### HCCS Scenario for Atlas A3 Training Products/Atlas A3 Inference Products

```c
// Create and initialize the communicator configuration item.
HcclCommConfig config;
HcclCommConfigInit(&config);
// Modify the communicator configuration as required.
config.hcclSymWinMaxMemSizePerRank = 10; //Unit: GB. Default value: 16. Set the total virtual symmetric memory size reserved by the current communicator to rankSize * config.hcclSymWinMaxMemSizePerRank;

// Obtain the communicator parameters.
uint32_t rankSize = 4;
uint32_t rankId = 0;
int32_t deviceId;
ACLCHECK(aclrtGetDevice(&deviceId));
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// Initialize the communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// Create a task stream.
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

// Configure physical memory properties.
aclrtPhysicalMemProp prop;
prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
prop.memAttr = ACL_HBM_MEM_HUGE;
prop.location.id = deviceId;
prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
prop.reserve = 0;

// Obtain the alignment granularity, usually 2 MB.
size_t granularity;
ACLCHECK(aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity));

// Align size by the granularity.
size_t size = 2 * 1024 * 1024;
size_t allocSize = (size + granularity - 1) / granularity * granularity;

// Reserve virtual memory.
void *virPtr;
ACLCHECK(aclrtReserveMemAddress(&virPtr, allocSize, 0, nullptr, 1));

// Allocate physical memory.
aclrtDrvMemHandle memHandle;
ACLCHECK(aclrtMallocPhysical(&memHandle, allocSize, &prop, 0));

// Establish the mapping from physical memory to virtual memory.
ACLCHECK(aclrtMapMem(virPtr, allocSize, 0, memHandle, 0));

size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
HcclCommSymWindow symWin;
// Register symmetric memory.
HCCLCHECK(HcclCommSymWinRegister(hcclComm, virPtr, sendBytes + recvBytes, &symWin, 1));

// Use symmetric memory.
void *sendBuff = virPtr;
void *recvBuff = static_cast<char*>(sendBuff) + sendBytes;

// Call the collective communication operator.
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// Block and wait until the collective communication tasks in the stream are complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Deregister the symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Free the memory.
ACLCHECK(aclrtUnmapMem(virPtr));
ACLCHECK(aclrtFreePhysical(memHandle));
ACLCHECK(aclrtReleaseMemAddress(virPtr));

// Destroy the stream.
ACLCHECK(aclrtDestroyStream(stream));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```

### Ascend 950PR/Ascend 950DT URMA Scenario

```c
// Create and initialize the communicator configuration item.
HcclCommConfig config;
HcclCommConfigInit(&config);

// Obtain the communicator parameters.
uint32_t rankSize = 4;
uint32_t rankId = 0;
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// Initialize the collective communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// Create a task stream.
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
size_t memSize = sendBytes + recvBytes;

// Allocate device memory.
void *devPtr = nullptr;
ACLCHECK(aclrtMalloc(&devPtr, memSize, ACL_MEM_MALLOC_HUGE_FIRST));

HcclCommSymWindow symWin;
// Register symmetric memory.
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// Use symmetric memory.
void *sendBuff = devPtr;
void *recvBuff = static_cast<char*>(sendBuff) + sendBytes;

// Call the collective communication operator.
// In the URMA scenario of Ascend 950PR/Ascend 950DT, HcclCommSymWinRegister only registers the symmetric memory window;
// When HcclAllGather internally creates UB/URMA communication channels, it completes memory registration, memHandle exchange, and remote memory information update through the HcclChannelAcquire-related process.
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// Wait for the collective communication tasks in the stream to complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Deregister the symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Free the memory.
ACLCHECK(aclrtFree(devPtr));

// Destroy the stream.
ACLCHECK(aclrtDestroyStream(stream));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
