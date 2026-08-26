# GetResByChannel

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:05:55.429Z pushedAt=2026-08-18T03:25:40.170Z -->

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

Obtains the handle to a channel-shared variable slot within a CCU kernel.

This slot is a shared variable that has already been reserved on the local end when the channel is established and is mirrored one-to-one with the peer's slot number. This API does not consume additional `Variable` resources; it only wraps the existing slot into an operable Variable object.

Typical scenario: the peer rank writes a value into shared variable slot N of the local channel through [WriteVariableWithNotify](../synchronization/WriteVariableWithNotify.md), and the local kernel uses `GetResByChannel<Variable>(ch, N)` to obtain the handle to the same slot and read the value sent by the peer end.

> [!NOTE] Note
> Currently, only the `Variable` specialization (`GetResByChannel<Variable>`) is supported. Template instantiation for other types causes a compile-time error.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
template <typename T>
T GetResByChannel(ChannelHandle channel, uint32_t index);

// Currently, only the variable specialization is supported:
template <>
Variable GetResByChannel<Variable>(ChannelHandle channel, uint32_t varIndex);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channel | Input | Cross-rank channel handle (`ChannelHandle`). The channel must have completed link establishment, and its pre-allocated variable pool must contain the slot corresponding to `varIndex`. |
| varIndex | Input | Variable index within the channel (starting from 0), corresponding to a slot in the variable array pre-allocated when the channel was established. |

## Return Value

Returns a `Variable` object whose `handle` points to the `varIndex`-th shared variable slot of the channel. This variable does not occupy additional `Variable` resources; its release right belongs to the channel and is not released along with the returned variable's destruction.

If the call fails, an exception is thrown (carrying an error code). Common error codes:

| Scenario | Error Code |
| --- | --- |
| `channel == nullptr`, invalid channel type, or no kernel currently in the registration state | `CCU_E_PTR` |
| `varIndex` out of range (exceeding the size of the channel's pre-allocated variable pool) | `CCU_E_PARA` |

## Constraints

- This API can only be called during the kernel registration phase, and must be within the kernel function body executed by `HcommCcuKernelRegister`.
- It does not additionally occupy new `Variable` resources. Its allocation behavior is completely different from that of `Variable v;` (ordinary construction), and the two are not interchangeable.
- The lifetime of the returned variable is managed by the channel and remains valid until the channel is destroyed.
- Currently only `T = Variable` is supported. Calling with other types results in a compile-time failure.

## Example

```cpp
using namespace AscendC::ccu;

// End-to-end scenario: the peer writes a value to shared variable slot 0 of the local channel through WriteVariableWithNotify.
// The local kernel reads the value through GetResByChannel.

CcuResult MyKernel(CcuKernelArg arg) {
    auto* params = static_cast<MyKernelArg*>(arg);
    ChannelHandle ch = params->channelHandle;

    // Obtain the pre-allocated Variable[0] of the channel (without allocating new variable resources).
    Variable syncVar = GetResByChannel<Variable>(ch, 0);

    // Wait for the peer to write (for details, see WriteVariableWithNotify).
    NotifyWait(ch, /*localNotifyIdx=*/0);

    // At this point, syncVar already holds the value written by the peer end and can be read for subsequent computation.
    Variable result;
    result = syncVar;

    return CCU_SUCCESS;
}
```
