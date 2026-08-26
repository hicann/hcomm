# HcclCommWorkingDevNicSet

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T08:42:04.220Z pushedAt=2026-08-15T06:26:29.767Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Not supported
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

In cluster scenarios, configures the communication NICs in a communicator. It supports switching communication between a device NIC and a backup NIC on the same NPU, where the backup NIC is the NIC of another die on the same NPU.

When configuring communication NICs in a single operation, all devices in the same communicator must call this API, and the ranks, useBackup, and nRanks parameter configurations delivered by all devices must be consistent.

> [!NOTE] Note
> When configuring communication NICs, note the impact of the following operations:
>
> - When commands for switching to the backup NIC and switching to the primary NIC are delivered at the same time, if one end of a link switches to the primary NIC while the other end switches to the backup NIC, the command execution fails.
> - If a link exists between device 1 and device 2, the first command switches device 1 to the backup NIC, and the second command switches device 2 to the default NIC, the link between device 1 and device 2 uses the default NIC for communication, while the links between device 1 and other devices use the backup NIC for communication.
> - If the NICs of device 1 and device 2 are backup NICs for each other, and both devices deliver the command to switch to the backup NIC at the same time, the communication NICs used by device 1 and device 2 are swapped.

## Function Prototype

```c
HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator for which the communication NIC is specified.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| ranks | Input | Array consisting of the IDs of the ranks whose communication NICs are to be specified in the communicator. |
| useBackup | Input | Whether the devices in ranks use the backup NIC. |
| nRanks | Input | Number of ranks whose NICs are to be switched. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- Before configuration, ensure that the communication tasks in the current environment are complete.
- When ranks in the same communicator call this API, the lengths of the ranks and useBackup arrays passed in must be consistent with the nRanks value.
- The actual NIC switchover occurs only when both of the following conditions are met.
  - Re-execution is enabled for inter-SuperPoD communication, that is, the "L2" configuration of the environment variable HCCL_OP_RETRY_ENABLE is set to 1.
  - The topology contains RDMA links.

- In unsupported scenarios, for example, when re-execution is not enabled for HCCL_OP_RETRY_ENABLE, HCCL_SUCCESS is returned and a WARNING is printed in the log, but the NIC is not actually switched.
- For the same rank, the HcclCommWorkingDevNicSet API must be called one by one in order. Concurrent call is not supported.
- For the entire communicator, when calling HcclCommWorkingDevNicSet, ensure that the same delivery order is used across different ranks.
- The communicator pointed to by the comm handle must have had operators delivered before NIC configuration can be performed. Subsequently delivered operators of the same type and with the same parameters will continue to use the same NIC configuration, while other newly delivered operators will use the default NIC for communication. If these requirements are not met, NIC configuration will fail.

## Example

```c
HcclComm comm;
uint32_t rankSize = 4;
uint32_t rankIds[rankSize] = {0, 3, 7, 12};
bool useBackup[rankSize] = {true, true, true, true};
HcclCommWorkingDevNicSet(comm, rankIds, useBackup, rankSize);
```
