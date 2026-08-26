# HcclGetHeterogMode

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T09:35:34.466Z pushedAt=2026-08-17T07:52:35.187Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Not supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Given a communicator, obtains the heterogeneous networking mode of the communicator.

## Function Prototype

```c
HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode *mode)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| mode | Output | Heterogeneous mode.<br>For details about the HcclHeterogMode type, see [HcclHeterogMode](../../datatype_definition/HcclHeterogMode.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

None

## Example

```c
HcclHeterogMode mode;
HcclResult ret = HcclGetHeterogMode(comm, &mode);
if (ret == HCCL_SUCCESS) {
    switch (mode) {
        case HCCL_HETEROG_MODE_HOMOGENEOUS:
            printf("Homogeneous networking\n");
            break;
        case HCCL_HETEROG_MODE_MIX_A2_A3:
            printf("A2/A3 heterogeneous networking\n");
            break;
        default:
            printf("Unknown networking mode\n");
            break;
    }
}
```
