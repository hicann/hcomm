# Introduction

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:08:39.758Z pushedAt=2026-08-18T03:41:13.918Z -->

This section provides the APIs for requesting virtual resource handles in the CCU kernel, as well as the APIs for assigning values to and performing arithmetic operations on `Variable`/`Address`.

CCU resource allocation adopts a two-phase "virtual first, physical later" model: during the kernel registration phase (inside the kernel function body), calling constructors such as `Variable()`/`Address()`/`Event()` only produces virtual handles (consuming no hardware resources and always succeeding), while the actual physical resource allocation is completed during the `HcommCcuKernelRegister` phase (after the kernel function finishes execution). All C++ wrapper classes follow the semantics of virtual allocation upon construction and no release upon destruction; physical resources are managed uniformly over the lifecycle of the CCU instance and are not released by C++ destructors.

Resources are classified into the following types:

| Resource Type | Unit Allocation | Batch Allocation | Channel Reference |
| --- | --- | --- | --- |
| Scalar value (`Variable`) | [Variable](Variable.md) | [Array\<Variable\>](Array.md) | [GetResByChannel](GetResByChannel.md) |
| Address (`Address`) | [Address](Address.md) | — | — |
| Completion event (`Event`) | [Event](Event.md) | [Array\<Event\>](Array.md) | — |
| On-chip buffer (`CcuBuffer`, 4 KB) | [CcuBuffer](CcuBuffer.md) | [Array\<CcuBuffer\>](Array.md) | — |
| Local on-chip memory composite address (`Address` + `Variable`) | [LocalAddr](LocalAddr.md) | — | — |
| Remote on-chip memory composite address (`Address` + `Variable`) | [RemoteAddr](RemoteAddr.md) | — | — |

In addition to resource allocation, [Variable](Variable.md) and [Address](Address.md) also provide assignment and arithmetic operators. These operators describe operations executed on the device side, operating on the corresponding `Variable`/`Address` objects during hardware execution, rather than being computed immediately on the host side.

## API List

- [Variable](Variable.md)
- [Address](Address.md)
- [Event](Event.md)
- [CcuBuffer](CcuBuffer.md)
- [LocalAddr](LocalAddr.md)
- [RemoteAddr](RemoteAddr.md)
- [Array](Array.md)
- [GetResByChannel](GetResByChannel.md)
