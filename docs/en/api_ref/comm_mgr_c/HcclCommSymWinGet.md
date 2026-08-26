# HcclCommSymWinGet

<!-- md-trans-meta sourceCommit=4296112684f605f4a436db49d4fc4ee45c3b6646 translatedAt=2026-08-14T08:41:04.072Z pushedAt=2026-08-15T01:47:03.384Z -->

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

Based on the address pointer of the registered symmetric memory, returns the corresponding window resource handle and its offset within the window.

For Atlas A3 training products/Atlas A3 inference products, this API supports the HCCS link communication scenario. For Ascend 950PR/Ascend 950DT, this API supports the URMA scenario.

## Function Prototype

```c
HcclResult HcclCommSymWinGet(HcclComm comm, void *ptr, size_t size, HcclCommSymWindow *winHandle, size_t *offset)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. |
| ptr | Input | Address pointer of the registered symmetric memory. The memory must have been registered by calling [HcclCommSymWinRegister](HcclCommSymWinRegister.md).<br>In the HCCS scenario of Atlas A3 training products/Atlas A3 inference products, this address is a virtual address that has been reserved and mapped to physical memory.<br>In the URMA scenario of Ascend 950PR/Ascend 950DT, this address is the registered device memory address. |
| size | Input | Size of the symmetric memory window.<br>Assume that the symmetric memory window size is symSize and the address pointer of the registered symmetric memory is addr. size must meet the following conditions:<br>  - size > 0<br>  - ptr+size <= addr + symSize |
| winHandle | Output | Pointer to the symmetric memory window resource handle. |
| offset | Output | Pointer to the offset.<br>Assume that the address pointer of the registered symmetric memory is addr. Then *offset = ptr - addr. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- For Atlas A3 training products/Atlas A3 inference products, only the HCCS link communication scenario is supported. For Ascend 950PR/Ascend 950DT, only the URMA scenario is supported.
- Only the scenario where the communication operator expansion mode is AI CPU is supported.

## Example

### HCCS scenario for Atlas A3 training products/Atlas A3 inference products

```c
// Create and initialize the communicator configuration.
HcclCommConfig config;
HcclCommConfigInit(&config);
// Modify the communicator configuration as required
config.hcclSymWinMaxMemSizePerRank = 10; // Unit: GB, default value: 16

// Obtain the communicator parameters
uint32_t rankSize = 4;
uint32_t rankId = 0;
int32_t deviceId;
ACLCHECK(aclrtGetDevice(&deviceId));
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// Initialize the collective communicator.
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

HcclCommSymWindow symWin;
// Register symmetric memory.
HCCLCHECK(HcclCommSymWinRegister(hcclComm, virPtr, allocSize, &symWin, 1));

// Use HcclCommSymWinGet to obtain symmetric memory resources.
HcclCommSymWindow tempWin;
size_t offset = 0;
HCCLCHECK(HcclCommSymWinGet(hcclComm, virPtr, allocSize, &tempWin, &offset));

// Deregister the symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Release the memory.
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
// Create and initialize the communicator configuration.
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

size_t memSize = 2 * 1024 * 1024;

// Allocate device memory.
void *devPtr = nullptr;
ACLCHECK(aclrtMalloc(&devPtr, memSize, ACL_MEM_MALLOC_HUGE_FIRST));

HcclCommSymWindow symWin;
// Register the symmetric memory.
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// Obtain the symmetric memory resource using HcclCommSymWinGet.
HcclCommSymWindow tempWin;
size_t offset = 0;
HCCLCHECK(HcclCommSymWinGet(hcclComm, devPtr, memSize, &tempWin, &offset));

// Deregister the symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Release the memory.
ACLCHECK(aclrtFree(devPtr));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
