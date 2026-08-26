# ReadReduce

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:50:49.151Z pushedAt=2026-08-17T11:16:03.193Z -->

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

Initiates a cross-rank read and in-place reduction operation in a CCU kernel. It reads data from the remote on-chip memory through the established `ChannelHandle`, and merges the data with the existing data in the local on-chip memory using the specified operator (`local = reduce(local, fetch_from_remote, opType)`). When the hardware completes the operation, bit `mask` of `event` is automatically set to 1. This API is asynchronous.

> [!NOTE] Note
> The parameter order follows the "destination first, source last" convention: `ReadReduce(ch, local, remote, ...)`, where `local` (the destination, which is also the accumulation end of the reduction) is in the second position, and `remote` (the source) is in the third position.

<!-- -->

> [!CAUTION] Caution
> All `ChannelHandle` objects used in the same kernel must belong to the same die. This API does not perform die verification at the call site, so it does not fail due to die inconsistency (the call site may still return `CCU_E_PTR` because no kernel is being registered, or `CCU_E_NOT_FOUND` because the handle is invalid; for details, see the return value table). Die consistency is uniformly verified by `HcommCcuKernelRegister` (based on all channels used during this kernel). If they are inconsistent, `HcommCcuKernelRegister` returns `CCU_E_PARA` instead of the return value of this API.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult ReadReduce(ChannelHandle ch, LocalAddr local, RemoteAddr remote,
                     Variable len, HcclDataType dataType, HcclReduceOp opType,
                     Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ch | Input | Cross-rank channel handle (`ChannelHandle`). The die bound to the channel must belong to the same die as all channels in this kernel (uniformly verified in `HcommCcuKernelRegister`; for details, see the caution above). |
| local | Input/Output | Destination address in the local on-chip memory (`LocalAddr`). A valid initial value must be written before the call; after the hardware completes, it is updated to the reduction result. |
| remote | Input | Source address in the remote on-chip memory (`RemoteAddr`). |
| len | Input | Number of bytes to operate on, of the `Variable` type (variable length at runtime). |
| dataType | Input | Data type. For values, see the `HcclDataType` enumeration. Only the following six types are supported: `HCCL_DATA_TYPE_UINT8`, `HCCL_DATA_TYPE_INT16`, `HCCL_DATA_TYPE_INT32`, `HCCL_DATA_TYPE_FP16`, `HCCL_DATA_TYPE_FP32`, and `HCCL_DATA_TYPE_BFP16`. Other values are rejected and an exception is thrown (with an error code). |
| opType | Input | Reduction operator. For values, see the `HcclReduceOp` enumeration. Only `HCCL_REDUCE_SUM` (sum), `HCCL_REDUCE_MAX` (maximum), and `HCCL_REDUCE_MIN` (minimum) are supported. `HCCL_REDUCE_PROD` is not supported; passing it is rejected and an exception is thrown (with an error code). When the SUM operation is used, the sum of low-precision input data is first promoted to a higher precision and then adjusted back to the same precision as the input data. |
| event | Input | Completion event object. When the hardware reduction read completes, `event[mask]` is automatically set. Downstream calls `EventWait(event, mask)` to wait. |
| mask | Input | 16-bit event mask. The default value is `1` (that is, bit0). |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | No kernel is currently being registered (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `local`/`remote`/`len`/`event` handle is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`; channel die inconsistency is reported by `HcommCcuKernelRegister` with `CCU_E_PARA`. If `dataType`/`opType` is outside the supported range (see the parameter description), an exception (carrying an error code) is thrown at runtime.

## Constraints

- `local` must have a valid initial value (such as 0 or negative infinity) written before the call; otherwise, the reduction result is undefined.
- The parameter order is local end (destination) first and remote end (source) second: `ReadReduce(ch, local, remote, ...)`.
- All `ChannelHandle` objects in the same kernel must belong to the same die. Channels of different dies cannot be mixed in the same kernel.
- This API is asynchronous. You must wait for the reduction to complete by calling `EventWait(event, mask)` before reading the result.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Read data from the remote on-chip memory and perform FP16 SUM reduction with the local data.
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    ChannelHandle ch = params->channelHandle;
    LocalAddr dst;      // dst must be pre-initialized with an initial value (for example, 0.0).
    RemoteAddr remote;
    Variable len;
    Event evt;

    ReadReduce(ch, dst, remote, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
