# Introduction

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T09:22:27.699Z pushedAt=2026-08-17T06:17:42.783Z -->

This section contains the host-side lifecycle management APIs for CCU kernels and the memory token query APIs.

With these APIs, you can create and destroy CCU instances, register and translate kernels, launch and execute kernels, and convert process virtual addresses into CCU access tokens.

## Prerequisites

The APIs in this section require the `CcuInsHandle` (CCU instance handle). Obtain it as follows:

1. **Create an HcclComm communicator**: See [HcclCommInitClusterInfo](../../../comm_mgr_c/HcclCommInitClusterInfo.md)
   or [HcclCommInitRootInfo](../../../comm_mgr_c/HcclCommInitRootInfo.md).

2. **Query the CCU instance handle from the communicator**: Call `HcclCommQueryCcuIns` (declared in `include/hccl/hccl_ccu_res.h`):

    ```c
    extern HcclResult HcclCommQueryCcuIns(HcclComm comm,
        CcuInsHandle *insHandles, uint32_t *insNum);
    ```

   Typical usage:

    ```c
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryCcuIns(comm, &insHandle, &insNum);
    // The current implementation always returns one CCU instance. insNum != 1 is treated as a failure.
    if (ret != HCCL_SUCCESS || insNum != 1) {
        // Error handling
    }
    ```

> This API belongs to the HCCL layer (not part of the `Hcomm*` / `Ccu*` series) and does not yet have a dedicated API reference page.
> For the complete signature, refer to the header file `include/hccl/hccl_ccu_res.h`.

## API Call Sequence

The standard sequence of API calls is as follows:

1. [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md): Starts a round of kernel registration.
2. [HcommCcuKernelRegister](HcommCcuKernelRegister.md): Registers a kernel function and records its operation sequence.
3. [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md): Ends the current round of registration, translates it into device instructions, and delivers them.
4. [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md): Starts kernel execution (can be called repeatedly).

Bypass APIs:

- [HcommCcuGetMemToken](HcommCcuGetMemToken.md): Converts a process virtual address into a memory token usable by the CCU. It is independent of the main process and can be called at any time.

## See Also

- [CCU Quick Start (Including the Complete AllGather Process)](../../../../comm_op_dev_guide/ccu_quick_start.md)
- [CCU Communication Operator Development Guide (Step-by-Step)](../../../../comm_op_dev_guide/ccu_comm_op_dev/README.md)
