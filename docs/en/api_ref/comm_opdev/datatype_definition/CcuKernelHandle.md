# CcuKernelHandle

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T10:45:10.290Z pushedAt=2026-08-18T11:45:46.900Z -->

## Description

CCU kernel handle, returned after registration by [HcommCcuKernelRegister](../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegister.md) and used to identify a registered kernel. The same handle can be launched multiple times through [HcommCcuKernelLaunch](../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelLaunch.md).

## Prototype

```c
typedef uint64_t CcuKernelHandle;
```
