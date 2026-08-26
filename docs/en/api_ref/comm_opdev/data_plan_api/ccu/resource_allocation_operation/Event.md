# Event

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:05:32.869Z pushedAt=2026-08-18T03:27:03.994Z -->

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

`ccu::Event` is a C++ wrapper class for completion events within a CCU kernel.

- Allocation on construction: the default constructor automatically allocates one `Event` virtual handle.
- No release on destruction: the destructor does not release hardware resources. The virtual handle becomes invalid after translation completes, and physical resources are managed and reclaimed uniformly over the lifecycle of the CCU instance.

Event is used to mark the completion status of asynchronous operations (such as data movement). The last parameter `(event, mask)` of the data movement API is automatically set by the hardware to `event[mask]` when the movement completes, and the downstream waits for that bit to be set through `EventWait(event, mask)`. Event itself does not hold a `mask` field; mask is passed independently on each `EventRecord`/`EventWait` call.

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class Event final {
public:
    Event();                    // Allocation on construction
    CcuEventHandle handle{0};  // Virtual handle
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `Event e;` | Allocates one `Event` virtual handle. It can be called only during the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |

> [!CAUTION] Caution
> The `Event` class has copy/move constructors, which only copy the `handle` field and do not allocate a new `Event`. After `Event e2 = e1;`, `e1` and `e2` point to the same `Event`, and `EventRecord`/`EventWait` on either object operates on the same hardware status bit. If an independent event is required, you must explicitly use the default constructor `Event e2;` (or use `Array<Event>`).

If construction fails, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code).

## Return Value

After successful construction, the `e.handle` field holds the allocated virtual handle. The handle value is allocated by the framework during the registration phase (the handle of the first event is `0`, and subsequent handles increment). Do not use "`handle != 0`" to determine validity.

## Constraints

- Event can only be constructed during the kernel registration phase.
- Destruction does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- Event has no `mask` field. The mask is passed as a parameter in `EventRecord`/`EventWait` and data movement APIs (the C++ default value is `1`). Different bits (masks) of the same event are independent of each other and can carry multiple pairs.
- The C++ construction form only allocates a virtual handle, always succeeds, and does not throw exceptions. When `Event` physical resources are insufficient, `CCU_E_UNAVAIL` is returned during the `HcommCcuKernelRegister` phase instead of being thrown at construction time. If multiple events are required, use [`Array<Event>`](Array.md) to allocate them in batches.

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Event evt;    // Obtain one event.

    // Used with the data movement API: the hardware automatically sets evt[0x1] when the movement is complete.
    LocalAddr src, dst;
    Variable len;
    LocalCopy(dst, src, len, evt);    // mask defaults to 0x1
    EventWait(evt);                    // Wait for bit0 to be set.

    return CCU_SUCCESS;
}
```
