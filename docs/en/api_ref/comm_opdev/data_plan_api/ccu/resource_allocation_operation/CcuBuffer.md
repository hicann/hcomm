# CcuBuffer

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:04:14.832Z pushedAt=2026-08-18T03:06:46.259Z -->

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

`ccu::CcuBuffer` is a C++ wrapper class for the on-chip buffer in a CCU kernel. Each buffer slice has a fixed size of 4096 bytes.

- Allocation upon construction: The default constructor automatically allocates one `CcuBuffer` virtual handle.
- No release upon destruction: The destructor does not release hardware resources. The virtual handle becomes invalid after translation is complete, and the physical resources are managed and reclaimed uniformly along with the CCU instance lifecycle.

A CcuBuffer slice is an on-chip high-speed buffer within the CCU die, used to transfer data between on-chip memory and the peer end, or as an operand for multi-buffer reduction (up to 8 buffers at a time).

> [!NOTE] Note
> The C++ class name retains the `Ccu` prefix (the class name is `CcuBuffer` instead of `Buffer`), and is written as `ccu::CcuBuffer` together with the namespace.

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class CcuBuffer final {
public:
    CcuBuffer();                      // Allocation upon construction
    CcuBufferHandle handle{0};       // Virtual handle
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `CcuBuffer buf;` | Allocates one 4 KB `CcuBuffer` virtual handle. It can be called only during the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |

> [!CAUTION] Caution
> The `CcuBuffer` class has copy/move constructors, which only copy the `handle` field and do not allocate a new `CcuBuffer`. After `CcuBuffer b2 = b1;`, `b1` and `b2` point to the same `CcuBuffer` slice. To obtain an independent CcuBuffer, you must explicitly use the default constructor `CcuBuffer b2;` (or use `Array<CcuBuffer>` to allocate multiple physically contiguous slices).

If construction fails, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code).

## Constraints

- CcuBuffer can be constructed only during the kernel registration phase.
- Destruction does not release hardware resources. Do not save or compare the `handle` value outside the kernel, because the handle becomes invalid after translation.
- Each CcuBuffer always represents a 4096-byte on-chip slice, and its size is not configurable.
- For APIs that operate on CcuBuffer (such as `LocalCopy`, `LocalReduce`, `Read`, and `Write`), the `len` passed in must not exceed 4096 bytes. The caller is responsible for ensuring this upper limit; exceeding it leads to undefined hardware behavior at runtime (such as data truncation or out-of-bounds access).
- The C++ construction only allocates a virtual handle and always succeeds without throwing an exception. When `CcuBuffer` physical resources are insufficient, `HcommCcuKernelRegister` returns `CCU_E_UNAVAIL` during the registration phase instead of throwing an exception at construction time. To reduce multiple buffers, you can use [`Array<CcuBuffer>`](Array.md) to allocate multiple physically contiguous buffers in a batch.

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    CcuBuffer buf;    // Allocate one 4 KB CcuBuffer.
    LocalAddr src;
    Variable len;
    Event evt;

    // Copy the on-chip memory data to CcuBuffer.
    LocalCopy(buf, src, len, evt);
    EventWait(evt);

    return CCU_SUCCESS;
}
```
