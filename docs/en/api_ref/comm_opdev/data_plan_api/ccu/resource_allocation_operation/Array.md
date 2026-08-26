# Array

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:03:23.434Z pushedAt=2026-08-18T03:07:48.843Z -->

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

`ccu::Array<T>` is a C++ template class for batch allocation of physically contiguous resources within a CCU kernel.

- Batch allocation upon construction: `count` physically contiguous virtual handles are allocated at once during construction.
- No release upon destruction: The destructor does not release hardware resources. Virtual handles become invalid after translation, and physical resources are managed and reclaimed uniformly over the CCU instance lifecycle.
- Move-only: copying is prohibited, while moving is allowed.

Physical contiguity is a prerequisite for certain APIs: `Load(addr, vArr, num)`/`Store` batch load/storage and `LocalReduce(buffers*, count, ...)` (2 ≤ count ≤ 8, for details, see [LocalReduce](../data_movement/LocalReduce.md)) all require the parameters to be physically contiguous, which must be requested through `Array<T>`. Declaring multiple objects separately does not guarantee physical contiguity.

Currently, only the following three specialized types are supported:

| Specialized Type | Resource Meaning |
| --- | --- |
| `Array<Variable>` | N physically contiguous `Variable` objects |
| `Array<Event>` | N physically contiguous `Event` objects |
| `Array<CcuBuffer>` | N physically contiguous `CcuBuffer` slices |

Instantiating `Array<T>` with other types will cause a compilation error (only the three specialized types above are supported).

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
template <typename T>  // T supports only Variable / Event / CcuBuffer.
class Array final {
public:
    explicit Array(uint32_t count);       // Batch virtual allocation upon construction; count can be 0
    T& operator[](uint32_t i);            // Subscript access (no bounds checking)
    const T& operator[](uint32_t i) const;
    T* data();                             // Obtain the pointer to the first element (for passing to batch APIs that require a pointer parameter).
    const T* data() const;
    uint32_t size() const;                 // Return the number of elements.
    // Copying is prohibited; moving is allowed.
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&& other) noexcept;
    Array& operator=(Array&& other) noexcept;
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `Array<Variable> vars(n);` | Allocates `n` physically contiguous `Variable` handles. |
| `Array<Event> evts(n);` | Allocates `n` physically contiguous `Event` handles. |
| `Array<CcuBuffer> bufs(n);` | Allocates `n` physically contiguous `CcuBuffer` handles. |

`count` can be 0 (`size() == 0`), in which case no hardware resources are allocated. If construction fails, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code).

## Member Function Description

| Function | Description |
| --- | --- |
| `arr[i]` | Returns the reference to the `i`th element (starting from 0, without bounds checking). |
| `arr.data()` | Returns the pointer to the first element, used to pass to batch APIs that require a `T*` parameter (for example, `LocalReduce(bufs.data(), count, ...)`). |
| `arr.size()` | Returns the `count` value at the time of allocation. |

## Constraints

- Array can be constructed only during the kernel registration phase.
- The destructor does not release hardware resources. Do not save the `handle` value of an element outside the kernel, because the handle becomes invalid after translation is complete.
- `Array<T>` specializes only the three types `Variable/Event/CcuBuffer`. Instantiating it with other types fails at compilation time.
- Multiple separately declared `Variable`/`Event`/`CcuBuffer` objects are not guaranteed to be physically contiguous, and cannot be used with APIs that require physically contiguous resources (such as batch `Load`/`Store` and multi-buffer `LocalReduce`).

> [!NOTE] Note
> `Load`/`Store` performs a contiguity check on the Variable array (returning `CCU_E_PARA` if it is not contiguous), whereas the multi-buffer overload of `LocalReduce` does not check whether the CcuBuffers are physically contiguous, so the caller must guarantee this. When multiple buffers are allocated without using `Array`, no error is reported immediately, but the runtime behavior is undefined.

- The C++ constructor only allocates virtual handles and always succeeds. If the resource pool cannot provide N contiguous physical resources, `CCU_E_UNAVAIL` is returned during the `HcommCcuKernelRegister` phase, rather than being thrown at construction time.

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    // Allocate four physically contiguous variables in batches for batch load.
    Array<Variable> vArr(4);
    Load(0x80000000ULL, vArr, 4);    // Load four uint64_t values at a time.

    // Allocate four physically contiguous CcuBuffers in batches for multi-buffer reduction.
    Array<CcuBuffer> bufs(4);
    Variable len;
    Event evt;
    len = 4096;
    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP16,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);

    // Access a single element by subscript.
    vArr[0] = 1024;    // Assign an immediate value to the 0th Variable.

    return CCU_SUCCESS;
}
```
