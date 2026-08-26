# EventRecord

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:10:56.065Z pushedAt=2026-08-18T06:15:49.425Z -->

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

Explicitly marks the specified mask bit of a local event as complete within a CCU kernel.

> [!NOTE] Note
> In most scenarios, you do not need to explicitly call this API. The `(event, mask)` parameter at the end of the [data movement API](../data_movement/README.md) (such as `LocalCopy`, `Read`, and `Write`) is automatically set by the hardware when the data movement is complete, which is equivalent to an implicit `EventRecord`. You need to explicitly call this API only when you need to use the end of a control flow branch or a position without data movement operations as a synchronization point.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult EventRecord(Event e, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| e | Input | Local event object. When the `Event` class is constructed, an `Event` virtual handle is allocated (physical resources are allocated in the `HcommCcuKernelRegister` phase), and no manual allocation is required. |
| mask | Input | 16-bit event mask that specifies the bit to be set. The default value is `1` (that is, bit0). Different bits of the same `Event` object are independent of each other and can carry multiple pairs. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | No kernel is currently being registered (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The `e` handle passed in is not registered in the current kernel. |
| `CCU_E_NOT_SUPPORT` | The API is called inside a hardware loop body, which is not supported. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`.

## Constraints

- This API cannot be called inside a hardware loop body; otherwise, `CCU_E_NOT_SUPPORT` is returned. The loop body is expanded into N parallel copies by the hardware, so the semantics of `EventRecord` are not unique in a parallel environment.
- Each `EventRecord` must be followed by a corresponding `EventWait`; otherwise, the waiting side will be blocked permanently, causing a hardware-level deadlock.
- The bit specified by `mask` must be consistent with the `mask` of `EventWait`; otherwise, `EventWait` will never receive the signal.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Manually insert a synchronization point where there is no data movement.
// Inside the CCU kernel function body
CcuResult MyKernel(CcuKernelArg arg) {
    Event evt;

    // After performing some operations, manually mark bit0 as complete.
    EventRecord(evt, 0x1);

    // The downstream waits for bit0 to be set.
    EventWait(evt, 0x1);
    return CCU_SUCCESS;
}
```
