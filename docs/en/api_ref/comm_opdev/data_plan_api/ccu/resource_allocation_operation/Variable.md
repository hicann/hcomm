# Variable

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:09:38.869Z pushedAt=2026-08-18T06:17:20.787Z -->

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

`ccu::Variable` is a C++ wrapper class for scalar values within a CCU kernel.

- Allocation on construction: the default constructor automatically allocates one `Variable` virtual handle.
- No release on destruction: the destructor does not release hardware resources; the virtual handle becomes invalid after translation completes, and the physical resource is managed and reclaimed uniformly over the lifecycle of the CCU instance.
- Operators are device operations: the assignment and arithmetic operators on `Variable` describe operations executed on the device side (hardware), operating on the corresponding `Variable` object at runtime, rather than being computed immediately on the host side.

> [!NOTE] Note
> CCU resource allocation adopts a two-phase "virtual first, physical second" model: the `Variable()` construction in the registration phase only produces a virtual handle, and the actual `Variable` resource allocation is completed in the `HcommCcuKernelRegister` phase (after the kernel function finishes execution).

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class Variable final {
public:
    Variable();                                                   // Allocation on construction
    void operator=(uint64_t immediate) const;                    // Assign an immediate value.
    void operator=(const Variable& other) const;                 // Assign between Variables.
    void operator+=(const Variable& other) const;               // In-place addition
    /*Internal expression object*/ operator+(const Variable& that) const;    // Addition (expression template)
    CondExpr operator==(uint64_t immediate);                     // Produce CondExpr.
    CondExpr operator!=(uint64_t immediate);                     // Produce CondExpr.
    CcuVariableHandle handle{0};                                 // Virtual handle.
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `Variable v;` | Allocate one `Variable` virtual handle. It can be called only in the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |

The `Variable` class has copy/move constructors (`Variable(const Variable&)` / `Variable(Variable&&)`). They only copy the `handle` field and do not allocate a new `Variable`; that is, after `Variable v2 = v1;`, `v1` and `v2` point to the same runtime scalar value, and assignment/arithmetic operations on either object operate on the same `Variable`. To obtain an independent `Variable`, you must explicitly use the default constructor `Variable v2;`. Similarly, the move assignment `v2 = std::move(v1)` is also only a handle copy (it does not strip the source handle).

This is completely different from `operator=(const Variable&)`, which delivers a device-side assignment instruction and performs a value transfer between two independent `Variable` objects.

If construction fails, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code).

## Operator Description

### Assignment Operators

| Expression Syntax | Hardware Semantics |
| --- | --- |
| `v = imm;` (`imm` is `uint64_t`) | `v ← imm`. The immediate value is determined during the registration phase and is immutable at runtime. |
| `d = s;` (`s` is `Variable`) | `d ← s`. Device-side assignment, rather than host-side handle copy. |

### Arithmetic Operators

| Expression Syntax | Hardware Semantics |
| --- | --- |
| `r = a + b;` | `r ← a + b` (a single dual-source addition instruction). `operator+` returns an expression template object, which is consumed by `operator=` to generate one device-side addition without creating a temporary `Variable`. |
| `r += b;` | `r ← r + b`. It is semantically identical to `r = r + b`, with no temporary object. |

> [!CAUTION] Caution
> `r = a + b` uses an expression template (internal type) to avoid creating a temporary `Variable` that would consume additional `Variable` resources. Do not store the result of `a + b` in an ordinary C++ variable; otherwise, the corresponding device-side operation will not be generated.

### Conditional Operators

| Expression Syntax | Return Type | Description |
| --- | --- | --- |
| `n == imm` | `CondExpr` | Produces a conditional expression object without generating any device operation. It is intended exclusively for consumption by the `CCU_IF`/`CCU_WHILE`/`CCU_DO...CCU_WHILE()` macros. |
| `n != imm` | `CondExpr` | Same as above. |

> [!CAUTION] Caution
> `CondExpr` can only be consumed by control flow macros and cannot be used as an ordinary C++ Boolean expression (for example, `if (n == 0)`). Placing it in an ordinary `if` statement does not generate any CCU control flow, and the expression is discarded directly.

## Constraints

- A variable can only be constructed during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- Currently, only addition is supported. Subtraction, multiplication, and division are not supported.
- An immediate value cannot directly participate in arithmetic (`v + 1` is invalid). It must first be assigned to a variable (`one = 1;`) before participating in an operation (`v = v + one;`).
- Construction only requests a virtual handle (it does not consume `Variable` physical resources), and **construction always succeeds within the kernel registration phase**. If construction occurs outside the registration phase (not inside the kernel function body called by `HcommCcuKernelRegister`), the underlying `CcuVariableAlloc` cannot find the current kernel and throws a `CcuException` carrying `CCU_E_PTR`. Insufficient `Variable` physical resources are reported by returning `CCU_E_UNAVAIL` at the `HcommCcuKernelRegister` phase, not triggered at construction time. This constraint echoes the earlier statement that "Variable can only be constructed during the kernel registration phase" — the latter states the rule, while the former states the consequence of violating the rule (throwing `CCU_E_PTR`).

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Variable n, i, one, step;

    // Assign an immediate value (determined during the registration phase).
    n = 100;
    i = 0;
    one = 1;
    step = 8;

    // Assign between Variables.
    Variable cursor;
    cursor = i;           // cursor ← i

    // Arithmetic: expression template syntax (r = a + b), generating only one device addition
    Variable sum;
    sum = i + step;       // sum ← i + step

    // In-place addition
    i += one;             // i ← i + one

    // Conditional operation (consumed by the CCU_WHILE macro)
    CCU_WHILE(n != 0) {
        // ...
    }

    return CCU_SUCCESS;
}
```
