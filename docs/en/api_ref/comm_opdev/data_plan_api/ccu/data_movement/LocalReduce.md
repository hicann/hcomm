# LocalReduce

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T09:48:30.482Z pushedAt=2026-08-17T10:57:30.304Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Initiates a local reduction operation (asynchronous) within a CCU kernel, merging the source data with the destination memory or multiple CcuBuffers using the specified operator. When the hardware completes the operation, bit `mask` of `event` is automatically set to 1. The following two data paths are supported:

| Reload  | Data Path                                      | Reduction Mode                                                                                                                                                                                                                                                     |
| --- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reload 1 | Local on-chip memory (`src`) → local on-chip memory (`dst`)                 | `dst = reduce(dst, src, opType)`, with the same input and output types                                                                                                                                                                                                                |
| Reload 2 | N local CcuBuffers → `buffers[0]` (2 ≤ N ≤ 8) | `buffers[0] = reduce(buffers[0..N-1], opType)`, where the hardware reduces N CcuBuffers to `buffers[0]` in one pass. The input data type can be (`HCCL_DATA_TYPE_UINT8`/`HCCL_DATA_TYPE_INT16`/`HCCL_DATA_TYPE_INT32`/`HCCL_DATA_TYPE_FP16`/`HCCL_DATA_TYPE_BFP16`/`HCCL_DATA_TYPE_FP32`), and the output data type supports the same precision as the input, or precision expansion under `HCCL_REDUCE_SUM` (for details, see the `outputDataType` parameter description). |

> [!NOTE] Note
> This API is asynchronous. After calling it, you must wait for the reduction to complete by calling `EventWait(event, mask)`. Otherwise, the data in the destination memory is undefined. The reduction is an in-place operation. Before the call, `dst` (reload 1) or `buffers[0]` (reload 2) must already contain a valid initial value (such as 0 or negative infinity).

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Reload 1: local on-chip memory → local on-chip memory reduction
CcuResult LocalReduce(LocalAddr dst, LocalAddr src, Variable len,
                      HcclDataType dataType, HcclReduceOp opType,
                      Event event, uint16_t mask = 1);
// Reload 2: Reduce N local CcuBuffers to buffers[0] (2 ≤ count ≤ 8)
CcuResult LocalReduce(CcuBuffer* buffers, uint32_t count,
                      HcclDataType dataType, HcclDataType outputDataType,
                      HcclReduceOp opType,
                      Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

### Reload 1 Parameters

| Parameter | Input/Output | Description |
| -------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| dst | Input/Output | Destination on-chip memory address (`LocalAddr`). A valid initial value must be written before the call; after the hardware completes, it is updated to the reduction result. |
| src | Input | Source on-chip memory address (`LocalAddr`). |
| len | Input | Number of bytes to operate on, of the `Variable` type (variable length at runtime). |
| dataType | Input | Data type. See values in the `HcclDataType` enumeration. Only the following six types are supported: `HCCL_DATA_TYPE_UINT8`, `HCCL_DATA_TYPE_INT16`, `HCCL_DATA_TYPE_INT32`, `HCCL_DATA_TYPE_FP16`, `HCCL_DATA_TYPE_FP32`, and `HCCL_DATA_TYPE_BFP16`. Other values are rejected and an exception is thrown (carrying an error code). |
| opType | Input | Reduction operator. See values in the `HcclReduceOp` enumeration. Only `HCCL_REDUCE_SUM` (sum), `HCCL_REDUCE_MAX` (maximum), and `HCCL_REDUCE_MIN` (minimum) are supported. `HCCL_REDUCE_PROD` is not supported; passing it is rejected and an exception is thrown (carrying an error code). When the SUM operation is used, the sum result of low-precision input data is first promoted to a higher precision and then adjusted back to the same precision as the input data. |
| event | Input | Completion event object. When the hardware reduction completes, `event[mask]` is automatically set. |
| mask | Input | 16-bit event mask. The default value is `1` (that is, bit0). |

### Reload 2 Parameters

| Parameter            | Input/Output | Description                                                                                                                                                                                                                                                                                                       |
| -------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| buffers        | Input/Output | Start address of the CcuBuffer array (`CcuBuffer*`), which cannot be `nullptr`. You are advised to allocate it through `ccu::Array<CcuBuffer>` to ensure physical contiguity. `buffers[0]` is the output location of the reduction result and must be written with a valid initial value before the call. In the precision expansion scenario (low-precision input + SUM with upgraded-precision output, for example, when INT8 is upgraded to FP32, the output elements are 4x the input), the passed CcuBuffer array must cover both the input and the expanded output occupancy (for example, when two INT8 inputs are upgraded to FP32 output, four CcuBuffers must be reserved instead of two). Insufficient reservation causes hardware read/write out-of-bounds and undefined behavior.                                                          |
| count          | Input    | Number of buffers. The value range is `[2, 8]` (exceeding the upper limit throws an exception and carries an error code). `count == 0` is directly rejected and returns `CCU_E_PARA`; `count == 1` is not rejected but the hardware behavior is undefined. For the single-buffer scenario, use reload 1. `count` must equal the actual length of the `buffers` array.                                                                                                                                                         |
| dataType       | Input    | Input data type. For values, see the `HcclDataType` enumeration. Only the following six types are supported: `HCCL_DATA_TYPE_UINT8`, `HCCL_DATA_TYPE_INT16`, `HCCL_DATA_TYPE_INT32`, `HCCL_DATA_TYPE_FP16`, `HCCL_DATA_TYPE_FP32`, and `HCCL_DATA_TYPE_BFP16`. Other values are rejected and throw an exception (carrying an error code). |
| outputDataType | Input    | Output data type. For values, see the `HcclDataType` enumeration. Two combinations are supported: ① same precision - the value is the same as `dataType`; ② upgraded precision - supported only when `opType == HCCL_REDUCE_SUM`, which reduces low-precision inputs and then upgrades them to high-precision output (for example, `HCCL_DATA_TYPE_INT8` → `HCCL_DATA_TYPE_FP32`; see "Call Example - Scenario 2"). Other `dataType`/`outputDataType` combinations return `CCU_E_NOT_SUPPORT` (see the return value table). In the upgraded-precision scenario, `buffers` must reserve CcuBuffers according to the precision expansion ratio. For details, see the `buffers` parameter description. |
| opType         | Input    | Reduction operator. For values, see the `HcclReduceOp` enumeration. Only `HCCL_REDUCE_SUM`, `HCCL_REDUCE_MAX`, and `HCCL_REDUCE_MIN` are supported; `HCCL_REDUCE_PROD` is not supported and is rejected with an exception (carrying an error code) when passed.                                                                                                                                                                              |
| len            | Input    | Number of bytes of each buffer participating in the reduction. The type is `Variable` and cannot exceed the size of a single slice (4096 bytes).                                                                                                                                                                                         |
| event          | Input    | Completion event object.                                                                                                                                                                                                                                                                                                  |
| mask           | Input    | 16-bit event mask. The default value is `1` (that is, bit0).                                                                                                                                                                                                                                                                                  |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value                 | Description                                                                    |
| ------------------- | --------------------------------------------------------------------- |
| `CCU_SUCCESS`       | The operation is successful.                                                                 |
| `CCU_E_PARA`        | Parameter error, including `buffers` being `nullptr` or `count` being 0.                                 |
| `CCU_E_NOT_SUPPORT` | Reload 2 only: the `dataType`/`outputDataType` combination does not satisfy the upgraded-precision/same-precision constraint (for details, see the `outputDataType` description). |

> [!NOTE] Note
> If the values of `dataType`/`opType` are outside the supported range (for details, see the parameter description), an exception (carrying an error code) will be thrown during runtime, instead of reporting it through the return value.

## Constraints

- The reduction is an in-place operation. Before the call, `dst` (reload 1) or `buffers[0]` (reload 2) must already contain a valid initial value; otherwise, the reduction result is undefined.
- For reload 2, `buffers` must point to a physically contiguous `CcuBuffer` array. You are advised to allocate it using `ccu::Array<CcuBuffer>`.
- For reload 2, `count` is in the range `[2, 8]`. When `count > 8`, an exception is thrown (carrying an error code). When `count == 0`, `CCU_E_PARA` is returned directly. When `count == 1`, it is not rejected but the hardware behavior is undefined (use reload 1 instead).
- For reload 2, in the precision expansion scenario (low-precision input + SUM with promoted-precision output), `buffers` must reserve additional CcuBuffers according to the expansion ratio to accommodate the expanded output; otherwise, the hardware reads/writes out of bounds and the behavior is undefined.
- When `CcuBuffer` is involved (reload 2), `len` must not exceed the size of a single `CcuBuffer` (4096 bytes). This upper limit must be guaranteed by the caller; exceeding it causes undefined hardware behavior at runtime.
- This API is asynchronous. You must wait for the reduction to complete by calling `EventWait(event, mask)` before reading the result.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: local on-chip memory → local on-chip memory reduction (FP16 SUM)
CcuResult MyKernel(CcuKernelArg arg) {
    LocalAddr dst, src;
    Variable len;
    Event evt;

    // The initial value (for example, 0.0) must be written to dst in advance.
    LocalReduce(dst, src, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 2: four CcuBuffers are reduced to buffers[0], with INT8 input promoted to FP32 output (a valid "input ≠ output" combination;
//        INT8→FP32 = 4× expansion, so the buffers array reserves CcuBuffers according to the expansion ratio to accommodate the expanded output)
CcuResult MyKernel2(CcuKernelArg arg) {
    Array<CcuBuffer> bufs(4);    // 4 physically contiguous buffers
    Variable len;
    Event evt;

    // bufs[0] must be pre-initialized with an initial value.
    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 3: regular reduction with the same input and output type (FP16 SUM)
CcuResult MyKernel3(CcuKernelArg arg) {
    Array<CcuBuffer> bufs(4);
    Variable len;
    Event evt;

    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP16,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
