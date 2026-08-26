# WriteVariableWithNotify

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T10:16:17.035Z pushedAt=2026-08-18T07:30:22.567Z -->

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

In a CCU kernel, combines "writing a remote variable value" and "triggering a remote Notify signal" into a single atomic operation. The hardware guarantees the completion order of "writing the value first and then sending Notify".

> [!CAUTION] Caution
> Do not split this API into two independent calls of "writing a variable" and "NotifyRecord". The hardware layer does not guarantee the arrival order of two independent operations. When the consumer receives the Notify signal and then reads the remote variable, there is a risk of reading a stale value. Only this API can guarantee the order at the hardware layer.

The consumer side waits for the signal through [NotifyWait](NotifyWait.md), and then reads the newly written value through `AscendC::ccu::GetResByChannel<Variable>(channel, remoteVarIdx)`.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult WriteVariableWithNotify(ChannelHandle channel, Variable var,
                                  uint32_t remoteVarIdx, uint32_t remoteNotifyIdx,
                                  uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channel | Input | Communication channel handle, which is the channel resource obtained through the link establishment process. For the definition of the `ChannelHandle` type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). The die bound to channel must be the same as the current die of this kernel. |
| var | Input | Local variable object. Its current value will be written to the remote channel-bound variable slot. |
| remoteVarIdx | Input | Remote variable slot index (0-based), which is the number negotiated and agreed upon by both ends of the channel. The consumer side references the same slot through `GetResByChannel<Variable>(channel, remoteVarIdx)`. |
| remoteNotifyIdx | Input | Remote notify slot index (0-based), which is the number negotiated and agreed upon by both ends of the channel. It must be exactly the same as the `localNotifyIdx` value of `NotifyWait` on the consumer side. |
| mask | Input | 16-bit event mask, specifying the bit to be set. The default value is `1`. It must be consistent with the `mask` of `NotifyWait` on the consumer side. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_NOT_FOUND` | The variable handle corresponding to `var` is not registered in the current kernel (the handle is invalid). |
| `CCU_E_PTR` | No kernel is currently being registered (the API is called outside the kernel registration phase). |

> [!NOTE] Note
> This API does not validate channel/`remoteVarIdx`/`remoteNotifyIdx` at the call site. The consistency of the die bound to the channel is uniformly validated by `HcommCcuKernelRegister` (based on all channels used during this kernel), and `CCU_E_PARA` is returned on inconsistency. This API does not intercept calls inside the hardware loop body. Do not use it inside the loop body.

## Constraints

- All channels in this kernel must belong to the same die. This consistency is uniformly validated by `HcommCcuKernelRegister`, and `CCU_E_PARA` is returned on inconsistency.
- `remoteVarIdx` and `remoteNotifyIdx` are channel-level slot numbers agreed upon by both communication parties at the protocol layer. They are not automatically allocated by the framework.
- Do not split this API into separate variable write operations and `NotifyRecord` calls. Otherwise, the hardware cannot guarantee the order between the value write and the Notify arrival, and the consumer may read a stale value.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: The producer sends the progress value to the consumer, and the consumer reads it after receiving the notification.
// Inside the producer CCU kernel function body (die A)
CcuResult ProducerKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    Variable progress;
    progress = 100;  // Assign the immediate value 100.

    // Atomic operation: write remote variable slot 2 + notify bit0 of remote slot 3.
    WriteVariableWithNotify(params->channel, progress,
                            /*remoteVarIdx=*/2, /*remoteNotifyIdx=*/3, 0x1);
    return CCU_SUCCESS;
}

// Inside the consumer CCU kernel function body (die B)
CcuResult ConsumerKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first
    // Wait for the signal from die A.
    NotifyWait(params->channel, /*localNotifyIdx=*/3, 0x1);

    // Safely read the value written by die A (guaranteed to be written).
    Variable received = GetResByChannel<Variable>(params->channel, /*varIdx=*/2);
    return CCU_SUCCESS;
}
```
