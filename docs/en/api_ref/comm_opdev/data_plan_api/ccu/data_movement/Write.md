# Write

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T09:52:03.734Z pushedAt=2026-08-17T11:30:48.553Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->

## Description

Initiates a cross-rank write operation (asynchronous) within a CCU kernel. It writes local data to the peer on-chip memory through the established `ChannelHandle`. When the hardware completes the operation, bit `mask` of `event` is automatically set to 1. The following two source types are supported:

| Reload | Source | Destination |
| --- | --- | --- |
| Reload 1 | Local on-chip memory (`LocalAddr`) | Peer on-chip memory (`RemoteAddr`) |
| Reload 2 | Local CCU on-chip buffer (`CcuBuffer`) | Peer on-chip memory (`RemoteAddr`) |

Note:

- The parameter order follows the "destination first, source last" convention: `Write(ch, remote, local, ...)`, that is, `remote` (destination, peer end) is in the second position and `local` (source, local end) is in the third position. This is the reverse of the order in `Read`, where `local` comes first. Do not confuse the two. The C++ type system prevents parameter order errors at compilation time through the different types of `RemoteAddr` and `LocalAddr`.
- All `ChannelHandle` objects used in the same kernel must belong to the same die. This API does not perform die validation at the call site, so it does not fail due to die inconsistency (the call site may still return `CCU_E_PTR` because no kernel is being registered, or `CCU_E_NOT_FOUND` because the handle is invalid; for details, see the return value table). Die consistency is uniformly validated by `HcommCcuKernelRegister` (based on all channels used during this kernel). If they are inconsistent, `HcommCcuKernelRegister` returns `CCU_E_PARA` instead of this API's return value.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Reload 1: local on-chip memory → peer on-chip memory
CcuResult Write(ChannelHandle ch, RemoteAddr remote, LocalAddr local,
                Variable len, Event event, uint16_t mask = 1);
// Reload 2: local CcuBuffer → peer on-chip memory
CcuResult Write(ChannelHandle ch, RemoteAddr remote, CcuBuffer local,
                Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ch | Input | Cross-rank channel handle (`ChannelHandle`). The die bound to the channel must belong to the same die as all channels in the current kernel (uniformly verified in `HcommCcuKernelRegister`; for details, see the caution above). |
| remote | Input | Peer on-chip memory destination address (`RemoteAddr`, a composite object of the peer on-chip memory address and token). |
| local | Input | Local source address. Reload 1 is `LocalAddr` (a composite object of the local on-chip memory address and token); reload 2 is `CcuBuffer` (a local CCU on-chip buffer slice object, with a maximum of 4096 bytes per slice). |
| len | Input | Number of bytes to write, of the `Variable` type (variable length at runtime). For reload 2, it cannot exceed 4096 bytes. |
| event | Input | Completion event object. When the hardware write completes, `event[mask]` is automatically set, and the downstream calls `EventWait(event, mask)` to wait. |
| mask | Input | 16-bit event mask. The default value is `1` (that is, bit0). |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation succeeds. |
| `CCU_E_PTR` | No kernel is currently in the registration state (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `remote`/`local`/`len`/`event` handle is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`; channel die inconsistency is returned as `CCU_E_PARA` by `HcommCcuKernelRegister`. For details, see [Description](#description).

## Constraints

- The parameter order is peer end (destination) first and local end (source) second: `Write(ch, remote, local, ...)`.
- All `ChannelHandle` objects in the same kernel must belong to the same die. Channels of different dies cannot be mixed in the same kernel.
- In reload 2, `len` cannot exceed the size of a single `CcuBuffer` slice (4096 bytes). The caller must ensure this upper limit; exceeding it leads to undefined hardware behavior at runtime.
- This API is asynchronous. You must call `EventWait(event, mask)` to wait for the write to complete before the peer data is guaranteed to be visible.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: local on-chip memory → peer on-chip memory
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    ChannelHandle ch = params->channelHandle;
    RemoteAddr remote;
    LocalAddr src;
    Variable len;
    Event evt;

    Write(ch, remote, src, len, evt);    // remote comes first, local comes last.
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 2: local CcuBuffer → peer on-chip memory
CcuResult MyKernel2(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    ChannelHandle ch = params->channelHandle;
    RemoteAddr remote;
    CcuBuffer buf;
    Variable len;
    Event evt;

    Write(ch, remote, buf, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
