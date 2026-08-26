# RemoteAddr

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:08:40.661Z pushedAt=2026-08-18T03:49:49.459Z -->

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

`ccu::RemoteAddr` is a C++ wrapper class for the remote on-chip memory address within a CCU kernel. It is a composite object of "address (`Address`) + token (`Variable`)", and its structure mirrors that of [LocalAddr](LocalAddr.md).

- Allocation on construction: the default constructor allocates one `Address` (for `addr`) and one `Variable` (for `token`) at a time.
- No release on destruction: the destructor does not release hardware resources. The virtual handle becomes invalid after translation is complete, and the physical resources are managed and reclaimed uniformly over the lifecycle of the CCU instance.

`RemoteAddr` is dedicated to cross-rank read/write operations (`Read`/`Write`/`ReadReduce`/`WriteReduce`), carrying the physical address and security token of the target memory on the remote rank. `addr` and `token` must come from the remote rank and must not be mixed with the `LocalAddr` fields of the local rank.

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class RemoteAddr final {
public:
    RemoteAddr();                        // Allocation on construction (allocates both Address and Variable)
    Address addr;                        // Remote on-chip memory address field (Address object)
    Variable token;                      // Remote security token field (Variable object)
    CcuRemoteAddrHandle handle{0};      // Composite handle
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `RemoteAddr ra;` | Allocates one `Address` virtual handle and one `Variable` virtual handle at a time, and fills in the three handles `ra.handle`, `ra.addr.handle`, and `ra.token.handle`. |

The C++ constructor only allocates virtual handles: it always succeeds when called within the kernel registration phase; if it is not called within the kernel registration phase, an exception is thrown during construction (carrying error code `CCU_E_PTR`). When the physical resources of `Address`/`Variable` are insufficient, `CCU_E_UNAVAIL` is returned during the `HcommCcuKernelRegister` phase, rather than being thrown during construction.

> [!CAUTION] Caution
> The `RemoteAddr` class has copy/move constructors, which only copy the three fields `handle`, `addr.handle`, and `token.handle` and do not allocate new `Address`/`Variable` objects. After `RemoteAddr r2 = r1;`, `r1` and `r2` point to the same group of `Address`/`Variable` objects. `operator=(const RemoteAddr&)` executes a device-side assignment instruction (with the same semantics as `LocalAddr`). The two are asymmetric, so pay special attention when using them.

## Field Description

| Field | Type | Description |
| --- | --- | --- |
| `addr` | `Address` | Target address of the remote on-chip memory. It must be filled with the VA and token of the remote rank (the result of the remote `HcommCcuGetMemToken`). For operator semantics, see [Address](Address.md). |
| `token` | `Variable` | Remote security token value. It must be paired with `addr`, come from the remote rank, and must not be mixed with the local token. For operator semantics, see [Variable](Variable.md). |

## Constraints

> [!CAUTION] Caution
> `token` is security information. It must not be printed in host or device logs, and must not be passed across ranks in plaintext. The `addr` and `token` fields must be the values of the remote rank's target memory, and their source is completely different from that of the local `LocalAddr` field.

- RemoteAddr can only be constructed during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid after translation is complete.
- Do not call `Address()`/`Variable()` separately on the embedded `addr`/`token` fields to construct new objects — `RemoteAddr()` has already allocated all subfields at a time.
- `RemoteAddr` must be used together with `ChannelHandle` to perform cross-rank RDMA operations over an established channel. Using it without a channel results in undefined behavior.

## Example

```cpp
using namespace AscendC::ccu;

// The address and token of the remote rank must be obtained from the remote end through inter-process communication (for example, a message passing framework),
// and then passed into the kernel through kernelArg or taskArgs+LoadArg.

CcuResult MyKernel(CcuKernelArg arg) {
    auto* params = static_cast<MyKernelArg*>(arg);

    RemoteAddr remote;
    // Assign from the kernelArg in the registration phase (fixed as an immediate value).
    remote.addr = params->remoteAddr;
    remote.token = params->remoteToken;

    // Used with the cross-rank Read/Write APIs.
    ChannelHandle ch = params->channelHandle;
    LocalAddr local;
    local.addr = params->localAddr;
    local.token = params->localToken;
    Variable len;
    Event evt;
    len = 1024;

    Read(ch, local, remote, len, evt);   // Read from the remote on-chip memory to the local on-chip memory.
    EventWait(evt);

    return CCU_SUCCESS;
}
```
