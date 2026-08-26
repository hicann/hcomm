# CCU API Introduction

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T10:17:28.181Z pushedAt=2026-08-18T07:40:56.297Z -->

This section provides the data plane APIs within the Communication Compute Unit (CCU) kernel, which are used to describe communication and synchronization logic in the operator kernel function body.

CCU adopts a two-phase model of "host registration + device execution": you call the APIs described in this section within the kernel function body to describe the logic. The framework first records these calls during the registration phase and translates them into CCU device instructions when registration ends. These commands are finally executed by the CCU hardware.

The APIs are classified by function as follows:

- [Resource Creation and Operation](./resource_allocation_operation/README.md)
- [Parameter Load/Storage](./arg_load_store/README.md)
- [Data Movement](./data_movement/README.md)
- [Synchronization](./synchronization/README.md)
- [Flow Control](./execution_control/README.md)
