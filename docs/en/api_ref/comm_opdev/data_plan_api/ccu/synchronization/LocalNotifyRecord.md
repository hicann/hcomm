# LocalNotifyRecord

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T10:12:38.392Z pushedAt=2026-08-18T06:40:58.973Z -->

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

Sends a cross-kernel synchronization signal within the same die in a CCU kernel. It is a producer-side API. Pairing is identified by a string tag: the producer and consumer sides with the same tag are automatically bound to the same synchronization resource.

> [!NOTE] Note
> This API is implemented in C++ as the `AscendC::ccu::EventRecord(const char* notifyTag, uint16_t mask)` overload, which shares the same function name as `EventRecord(Event, mask)` and is distinguished by the input parameter type. It applies to sequential synchronization between different kernels within the same die (cross-kernel on the same device), and does not apply to cross-die or cross-rank scenarios. For cross-die scenarios, use [NotifyRecord](NotifyRecord.md).

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// LocalNotifyRecord corresponds to this overload.
CcuResult EventRecord(const char *notifyTag, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| notifyTag | Input | String tag. It is paired with the `notifyTag` of the consumer-side `LocalNotifyWait` as long as they are consistent, without the need to allocate a number in advance. It cannot be a null pointer. It must remain valid during kernel registration (it must be a string literal or `static` storage within the kernel), and a temporary array on the stack cannot be used. |
| mask | Input | 16-bit event mask that specifies the bit to be set. The default value is `1`. Different bits of the same tag are independent of each other. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | `notifyTag` is a null pointer, or no kernel is currently in the registration state. |
| `CCU_E_NOT_SUPPORT` | The API is called inside a hardware loop body, which is not supported. |

## Constraints

- The API cannot be called inside a hardware loop body; otherwise, `CCU_E_NOT_SUPPORT` is returned. The loop body is expanded into N copies in parallel by the hardware, and the semantics of the notify record are not unique in a parallel environment.
- `notifyTag` must remain valid during kernel registration. Use a string literal (such as `"phase1_done"`) or a `static char[]` inside the kernel. Do not use a temporary `char[]` on the stack.
- The producer-side `LocalNotifyRecord` and the consumer-side `LocalNotifyWait` must use exactly the same `notifyTag` string content for pairing.
- The same tag can carry multiple pairs, distinguished by different bits of `mask`. The `mask` on the waiting side must be consistent with the one here.
- `LocalNotifyRecord` and the corresponding `LocalNotifyWait` must appear in pairs, and `LocalNotifyWait` must be located after `LocalNotifyRecord`. Any unpaired `LocalNotifyWait` will be blocked permanently, causing a hardware-level deadlock.

## Example

> In the following example, `EventRecord("phase1_done", 0x1)` is the `LocalNotifyRecord` described in this document. At the C++ layer, it shares the same function name as [EventRecord](EventRecord.md) (the `Event` object overload), and the two are distinguished by the input parameter type (`const char *` vs `Event`).

```cpp
using namespace AscendC::ccu;

// Scenario: On the same die, PhaseProducerKernel notifies PhaseConsumerKernel to continue after completing the phase-1 operation.
// Inside the CCU kernel function body registered to core 0.
CcuResult PhaseProducerKernel(CcuKernelArg arg) {
    // ... Execute phase-1 operations ...

    // Notify PhaseConsumerKernel on the same die that phase 1 is complete (that is, LocalNotifyRecord semantics)
    EventRecord("phase1_done", 0x1);
    return CCU_SUCCESS;
}

// Registered in the CCU kernel function body on core 1.
CcuResult PhaseConsumerKernel(CcuKernelArg arg) {
    // Wait for the phase 1 completion signal sent by PhaseProducerKernel.
    EventWait("phase1_done", 0x1);

    // The phase 1 results of PhaseProducerKernel can now be safely used.
    return CCU_SUCCESS;
}
```
