# Store

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T09:46:16.647Z pushedAt=2026-08-17T09:23:04.218Z -->

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

Writes the `uint64_t` value of one or more variables to an on-chip memory address in a CCU kernel.

It is the reverse operation of `Load` and supports the same two address types, which are automatically selected based on the type of the first parameter:

| Reload Group | Address Type | When Is the Address Determined |
| --- | --- | --- |
| Reload 1/2 | Immediate address (`uint64_t`) | Determined in the registration phase |
| Reload 3/4 | Variable address (`Variable`) | Read from a hardware register at runtime |

Within each group, the reloads are further divided by the number of stored elements into a single variable (num=1) and a batch `Array<Variable>` (num>1).

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Reload 1: Store one variable to an immediate address.
CcuResult Store(uint64_t addr, Variable v);
// Reload 2: Store num variables in batch to an immediate address.
CcuResult Store(uint64_t addr, Array<Variable>& vArr, uint32_t num);
// Reload 3: Store one variable to a variable address (indirect addressing).
CcuResult Store(Variable addrVar, Variable v);
// Reload 4: Store num variables in batch to a variable address (indirect addressing).
CcuResult Store(Variable addrVar, Array<Variable>& vArr, uint32_t num);
} // namespace ccu
} // namespace AscendC
```

## Parameters

### Reload 1/2 Parameters (Immediate Address)

| Parameter | Input/Output | Description |
| --- | --- | --- |
| addr | Input | Immediate on-chip memory destination address (`uint64_t`). It must be a physical address accessible to the CCU or a tokenized VA, determined in the registration phase and immutable at runtime. |
| v | Input | Source variable (reload 1, num=1). At runtime, the value of this variable is written to the location pointed to by addr in the on-chip memory. |
| vArr | Input | First element of the source variable array (reload 2, num>1). It must be allocated through `ccu::Array<Variable>` to ensure physical contiguity. |
| num | Input | Number of `uint64_t` elements to store (reload 2). It must be greater than 0.<br>Set the location pointed to by addr in the on-chip memory to `mem[addr]`. When `num>1`, `vArr[0], vArr[1], ..., vArr[num-1]` are written to `mem[addr], mem[addr+8], ..., mem[addr+(num-1)*8]`, respectively. |

### Reload 3/4 Parameters (Variable Address, Indirect Addressing)

| Parameter | Input/Output | Description |
| --- | --- | --- |
| addrVar | Input | Address variable (`Variable`). At runtime, the value stored in this variable is used as the on-chip memory target address, and a valid address value must have been assigned. |
| v | Input | Source variable (reload 3, num=1). |
| vArr | Input | First element of the source variable array (reload 4, num>1). It must be allocated through `ccu::Array<Variable>` to ensure physical contiguity. |
| num | Input | Number of `uint64_t` elements to store (reload 4), which must be greater than 0. The semantics are the same as reload 2, with the address taken from the runtime value of `addrVar`. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PARA` | Parameter error: `num` is 0; or when `num>1`, the elements of `vArr` are not physically contiguous. |
| `CCU_E_PTR` | No kernel is currently being registered (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `v`/`vArr` handle (or the adjacent handle of `vArr[1..num-1]`) is not registered in the current kernel; or `num` is greater than the actual length of `vArr`, causing access to a non-existent adjacent handle. |

## Constraints

- `addr` must be a physical address accessible to the CCU or a tokenized VA. Directly passing a non-tokenized process virtual address will trigger a driver error.
- `num` must be greater than 0. Passing 0 returns `CCU_E_PARA`.
- When `num>1` (reloads 2/4), `vArr` must point to a physically contiguous array of `Variable` objects, which must be allocated through `ccu::Array<Variable>`. Declaring multiple `Variable` objects separately does not guarantee physical contiguity. If this requirement is violated, `CCU_E_PARA` is returned at the `Store(...)` call site.
- `num` must be less than or equal to the actual length of `vArr`. This API does not check `num <= vArr.size()`. If `num` is out of range, adjacent handles that do not belong to `vArr` may be accessed, or `CCU_E_NOT_FOUND` is returned. Ensure that `num` does not exceed the allocated length.
- `addrVar` (reloads 3/4) must have been assigned a valid address value (through `LoadArg`, immediate assignment, or arithmetic operations) before this API is called.
- The immediate address (reloads 1/2) is determined during the registration phase and cannot be changed at runtime. Use reloads 3/4 when a runtime dynamic address is required.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: Store one variable to an immediate address (reload 1).
CcuResult MyKernel(CcuKernelArg arg) {
    Variable result;
    // ... perform computation on result ...
    Store(0x20000000ULL, result);
    return CCU_SUCCESS;
}

// Scenario 2: Store four variables in batches to an immediate address (reload 2, num>1).
CcuResult MyKernel2(CcuKernelArg arg) {
    Array<Variable> vArr(4);    // 4 physically contiguous variables
    // ... perform computation on vArr ...
    Store(0x80000000ULL, vArr, 4);
    return CCU_SUCCESS;
}

// Scenario 3: Store a variable to the address specified by a variable (reload 3).
CcuResult MyKernel3(CcuKernelArg arg) {
    Variable dstAddr, result;
    LoadArg(dstAddr, 0);    // Pass the destination address by the host.
    // ... perform computation on result ...
    Store(dstAddr, result); // Write through indirect addressing at runtime
    return CCU_SUCCESS;
}
```
