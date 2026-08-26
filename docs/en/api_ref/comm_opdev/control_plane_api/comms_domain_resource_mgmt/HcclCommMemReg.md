# HcclCommMemReg

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:26:39.719Z pushedAt=2026-08-17T06:44:03.958Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Registers the allocated memory with a communicator and obtains the corresponding registration handle.

## Function Prototype

```c
HcclResult HcclCommMemReg(HcclComm comm, const char *memTag, const CommMem *mem, HcclMemHandle *memHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| memTag | Input | Memory string tag. The maximum character length is HCCL_RES_TAG_MAX_LEN.<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| mem | Input | Memory information. For details about the CommMem type, see [CommMem](../../datatype_definition/CommMem.md). |
| memHandle | Output | Memory handle.<br>The HcclMemHandle type is defined as follows:<br>typedef void *HcclMemHandle; |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- Within a communicator, only one block of memory can be registered for the same memTag.
- Within a communicator, registering the same memTag repeatedly returns the HCCL_E_PARA error and does not reuse the existing registered memory handle.
- Within a communicator, different memTags can be mapped to overlapping or identical memory regions.

## Example

```c
HcclComm comm; // Communicator handle, omitted for brevity
const char* memTag = "memTag"; // Memory tag
void *deviceBuffer = nullptr; // Address of the allocated device memory (allocated through APIs such as aclrtMalloc)
CommMem memInfo; // Memory information
memInfo.addr = deviceBuffer; // Configure the address of the allocated memory
memInfo.size = 1024; // Configure the size of the allocated memory
memInfo.type = COMM_MEM_TYPE_DEVICE; // Configure the type of the allocated memory
HcclMemHandle memHandle; // Memory handle
HcclCommMemReg(comm, memTag, &memInfo, &memHandle);
```
