# WriteReduce

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:53:08.636Z pushedAt=2026-08-17T11:41:33.216Z -->

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

Initiates a cross-rank write and in-place reduction on the peer end within a CCU kernel. Through the established `ChannelHandle`, this API sends the local on-chip memory data to the peer end and merges it with the existing data in the peer on-chip memory using the specified operator (`remote = reduce(remote, local, opType)`). When the hardware completes the operation, bit `mask` of `event` is automatically set to 1. This API is asynchronous.

Note:

- The parameter order follows the "destination first, source second" convention: `WriteReduce(ch, remote, local, ...)`, where `remote` (the destination, the peer accumulator of the reduction) is in the second position and `local` (the source, the local end) is in the third position.
- All `ChannelHandle` objects used in the same kernel must belong to the same die. This API does not perform die verification at the call site, so it does not fail due to die inconsistency (the call site may still return `CCU_E_PTR` because no kernel is being registered, or `CCU_E_NOT_FOUND` because the handle is invalid; for details, see the return value table). Die consistency is uniformly verified by `HcommCcuKernelRegister` (based on all channels used during this kernel). If they are inconsistent, `HcommCcuKernelRegister` returns `CCU_E_PARA` instead of this API's return value.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult WriteReduce(ChannelHandle ch, RemoteAddr remote, LocalAddr local,
                      Variable len, HcclDataType dataType, HcclReduceOp opType,
                      Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ch | Input | Cross-rank channel handle (`ChannelHandle`). The die bound to the channel must belong to the same die as all channels in this kernel (uniformly verified in `HcommCcuKernelRegister`; for details, see the caution above). |
| remote | Input | Target address of the peer on-chip memory (`RemoteAddr`). The peer memory must have a valid initial value written before the call (written by the peer kernel); after the hardware completes, the content at this address is updated to the reduction result. |
| local | Input | Source address of the local on-chip memory (`LocalAddr`). |
| len | Input | Number of bytes to operate on, of type `Variable` (variable length at runtime). |
| dataType | Input | Data type. For values, see the `HcclDataType` enumeration. Only the following six types are supported: `HCCL_DATA_TYPE_UINT8`, `HCCL_DATA_TYPE_INT16`, `HCCL_DATA_TYPE_INT32`, `HCCL_DATA_TYPE_FP16`, `HCCL_DATA_TYPE_FP32`, and `HCCL_DATA_TYPE_BFP16`. Other values are rejected and an exception is thrown (with an error code). |
| opType | Input | Reduction operator. For values, see the `HcclReduceOp` enumeration. Only `HCCL_REDUCE_SUM` (sum), `HCCL_REDUCE_MAX` (maximum), and `HCCL_REDUCE_MIN` (minimum) are supported. `HCCL_REDUCE_PROD` is not supported; passing it is rejected and an exception is thrown (with an error code). When the SUM operation is used, the sum result of low-precision input data is first promoted to a higher precision and then adjusted back to the same precision as the input data. |
| event | Input | Completion event object. When the hardware reduction write completes, `event[mask]` is automatically set, and the downstream calls `EventWait(event, mask)` to wait. |
| mask | Input | 16-bit event mask. The default value is `1` (that is, bit0). |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation succeeded. |
| `CCU_E_PTR` | No kernel is currently in the registration state (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `remote`/`local`/`len`/`event` handle is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`; channel die inconsistency is returned by `HcommCcuKernelRegister` as `CCU_E_PARA`. If `dataType`/`opType` is outside the supported range (see the parameter description), an exception is thrown at runtime (with an error code).

## Constraints

- The peer `remote` memory must have a valid initial value written before the call (written by the peer kernel); otherwise, the reduction result is undefined.
- The parameter order is peer end (destination) first and local end (source) second: `WriteReduce(ch, remote, local, ...)`.
- All `ChannelHandle` objects in the same kernel must belong to the same die. Channels of different dies cannot be mixed in the same kernel.
- This API is asynchronous. You must call `EventWait(event, mask)` to wait for the reduction to complete, so that the remote data is guaranteed to be visible.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Write local FP16 data to the remote end and perform SUM reduction.
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    ChannelHandle ch = params->channelHandle;
    RemoteAddr remote;    // The remote end must write the initial value in advance.
    LocalAddr src;
    Variable len;
    Event evt;

    WriteReduce(ch, remote, src, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
