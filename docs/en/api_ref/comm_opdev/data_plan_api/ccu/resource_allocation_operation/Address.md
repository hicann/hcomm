# Address

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:03:00.138Z pushedAt=2026-08-18T02:45:19.572Z -->

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

`ccu::Address` is a C++ wrapper class for address values within a CCU kernel.

- Allocation upon construction: the default constructor automatically allocates one `Address` virtual handle.
- No release upon destruction: The destructor does not release hardware resources; the virtual handle becomes invalid after translation is complete, and physical resources are managed and reclaimed uniformly over the lifecycle of the CCU instance.
- Operators are device operations: The assignment and arithmetic operators on `Address` describe operations executed on the device (hardware), rather than immediate computation on the host; at runtime, they operate on the corresponding `Address` object.

Unlike `Variable` (a scalar value), `Address` specifically carries an address value (the physical address of on-chip memory). A typical usage is to add a base address and an offset to obtain the target address for use by data movement APIs.

> [!NOTE] Note
> `Address` carries a device-side address value and cannot be directly dereferenced on the host side. An Address cannot be assigned to a variable (no reverse assignment API is provided), but a variable can be assigned to an Address (`addr = var;`).

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class Address final {
public:
    Address();                                                   // Alloc upon construction
    void operator=(uint64_t immediate) const;                   // Assign an immediate address.
    void operator=(const Variable& var) const;                  // Assign the Variable value to Address.
    void operator=(const Address& other) const;                 // Assign between Address objects
    void operator+=(const Variable& var) const;                 // Add Variable to Address in place.
    void operator+=(const Address& other) const;               // Add Address to Address in place.
    /*Internal expression object*/ operator+(const Address& that) const;    // Address+Address
    /*Internal expression object*/ operator+(const Variable& var) const;    // Address+Variable
    CcuAddressHandle handle{0};                                 // Virtual handle
};
// Variable + Address (commutative law, global function)
/*Internal expression object*/ operator+(const Variable& var, const Address& addr);
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `Address a;` | Allocate one `Address` virtual handle. It can be called only during the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |

> [!CAUTION] Caution
> The `Address` class has copy/move constructors, which only copy the `handle` field and do not allocate a new `Address`. After `Address a2 = a1;`, `a1` and `a2` point to the same runtime address value. If an independent `Address` is required, you must explicitly use the default constructor `Address a2;`. `operator=(const Address&)` delivers a device-side assignment instruction, which moves values between two independent `Address` objects.

If construction fails, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code).

## Operator Description

### Assignment Operators

| Expression Syntax | Hardware Semantics |
| --- | --- |
| `addr = imm;` (`imm` is `uint64_t`) | `addr ← imm`. The address immediate is determined during the registration phase and is immutable at runtime. |
| `addr = var;` (`var` is `Variable`) | `addr ← var`. Writes the runtime value of the Variable into the `Address` object, suitable for runtime dynamic addresses. |
| `dst = src;` (`src` is `Address`) | `dst ← src`. Assignment between Address objects. |

### Arithmetic Operators

| Expression Syntax | Hardware Semantics |
| --- | --- |
| `r = a + b;` (both `a` and `b` are `Address`) | `r ← a + b`. |
| `r = addr + var;` / `r = var + addr;` | `r ← addr + var`. The two forms have the same semantics (commutative law). |
| `addr += var;` (`var` is a `Variable`) | `addr ← addr + var` (in-place operation, saving one instruction compared with `addr = addr + var`). |
| `addr += other;` (`other` is an `Address`) | `addr ← addr + other`. |

> [!NOTE] Note
> Both `r = addr + var` and `r = var + addr` are valid (the latter uses the global `operator+(Variable, Address)` friend function). The two forms have the same semantics (commutative law), and the parameter order is handled uniformly inside the operator overloading.

## Constraints

- An `Address` can only be constructed during the kernel registration phase.
- Destruction does not release hardware resources. Do not save or compare the `handle` value outside the kernel. The handle becomes invalid once translation is complete.
- `Address` cannot be assigned to `Variable`, that is, the `Variable = Address` operation is not provided.
- Currently, only addition is supported. Subtraction, multiplication, and division are not supported.
- An immediate value cannot directly participate in arithmetic (`addr + 0x100` is invalid). You must first assign the offset to a variable before using it in an operation.
- The C++ constructor only allocates a virtual handle and always succeeds without throwing an exception. When `Address` physical resources are insufficient, `HcommCcuKernelRegister` returns `CCU_E_UNAVAIL` during the registration phase instead of throwing an exception at construction time.

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Address base, dst;
    Variable offset, stride;

    // Assign an immediate address (fixed during the registration phase).
    base = 0x80000000ULL;

    // Assign a variable value to an Address (runtime dynamic address).
    // offset is injected through LoadArg at launch time.
    LoadArg(offset, 0);
    dst = offset;         // dst ← offset (determined at runtime)

    // Address + Variable offset (two equivalent forms)
    stride = 4096;
    dst = base + stride;  // r = addr + var
    dst = stride + base;  // r = var + addr (equivalent)

    // Offset Address in place
    base += stride;        // base += 4096

    // Address + Address
    Address result;
    result = base + dst;

    return CCU_SUCCESS;
}
```
