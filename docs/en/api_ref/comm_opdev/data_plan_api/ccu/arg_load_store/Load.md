# Load

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T09:45:41.149Z pushedAt=2026-08-17T01:50:06.195Z -->

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

Reads one or more `uint64_t` values from on-chip memory and writes them to a variable within a CCU kernel.

Two address types are supported, and the type is automatically selected based on the type of the first parameter:

| Reload Group | Address Type | When Is the Address Determined |
| --- | --- | --- |
| Reload 1/2 | Immediate address (`uint64_t`) | Determined during the registration |
| Reload 3/4 | Variable address (`Variable`) | Read from a hardware register at runtime |

Within each group, the reloads are further divided by the target quantity into a single-variable (num=1) reload and a batch `Array<Variable>` (num>1) reload.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Reload 1: Load one variable from an immediate address.
CcuResult Load(uint64_t addr, Variable v);
// Reload 2: Load num variables in batch from an immediate address.
CcuResult Load(uint64_t addr, Array<Variable>& vArr, uint32_t num);
// Reload 3: Load one variable from a variable address (indirect addressing).
CcuResult Load(Variable addrVar, Variable v);
// Reload 4: Load num variables in batch from a variable address (indirect addressing).
CcuResult Load(Variable addrVar, Array<Variable>& vArr, uint32_t num);
} // namespace ccu
} // namespace AscendC
```

## Parameters

### Reload 1/2 Parameters (Immediate Address)

| Parameter | Input/Output | Description |
| --- | --- | --- |
| addr | Input | Immediate on-chip memory address (`uint64_t`). It must be a physical address accessible to the CCU or a tokenized VA, determined in the registration phase and immutable at runtime. |
| v | Input/Output | Target variable (reload 1, num=1). At runtime, reads 8 bytes from the location pointed to by addr in on-chip memory and writes them to this variable. |
| vArr | Input/Output | First element of the target variable array (reload 2, num>1). It must be allocated through `ccu::Array<Variable>` to ensure physical contiguity. |
| num | Input | Number of `uint64_t` elements to load (reload 2). It must be greater than 0.<br>Let the location pointed to by addr in on-chip memory be `mem[addr]`. When `num>1`, reads `mem[addr], mem[addr+8], ..., mem[addr+(num-1)*8]` and writes them to `vArr[0], vArr[1], ..., vArr[num-1]`, respectively. |

### Parameters of Reloads 3/4 (Variable Address, Indirect Addressing)

| Parameter | Input/Output | Description |
| --- | --- | --- |
| addrVar | Input | Address variable (`Variable`). At runtime, the value stored in this variable is used as the on-chip memory address, and a valid address value must have been assigned to it. |
| v | Input/Output | Target variable (reload 3, num=1). |
| vArr | Input/Output | First element of the target variable array (reload 4, num>1). It must be allocated through `ccu::Array<Variable>` to ensure physical contiguity. |
| num | Input | Number of `uint64_t` elements to load (reload 4). It must be greater than 0. The semantics are the same as reload 2, with the address taken from the runtime value of `addrVar`. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PARA` | Parameter error: `num` is 0; or when `num>1`, the elements of `vArr` are not physically contiguous. |
| `CCU_E_PTR` | No kernel is currently in the registration phase (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `v`/`vArr` handle (or the adjacent handles of `vArr[1..num-1]`) is not registered in the current kernel; or when `num` is greater than the actual length of `vArr`, a nonexistent adjacent handle is accessed. |

## Constraints

- `addr` must be a physical address accessible to the CCU or a tokenized VA. Directly passing a process virtual address that has not been tokenized will trigger a driver error.
- `num` must be greater than 0. If 0 is passed in, `CCU_E_PARA` is returned.
- When `num>1` (reloads 2/4), `vArr` must point to a physically contiguous variable array, which must be allocated through `ccu::Array<Variable>`. Declaring multiple `Variable` objects separately does not guarantee physical contiguity. If this requirement is violated, `CCU_E_PARA` is returned at the `Load(...)` call site.
- `num` must be ≤ the actual length of `vArr`. This API does not verify `num <= vArr.size()`. If `num` is out of bounds, adjacent handles that do not belong to `vArr` are accessed or `CCU_E_NOT_FOUND` is returned. Ensure that `num` does not exceed the allocated length.
- `addrVar` (reloads 3/4) must have been assigned a valid address value (through `LoadArg`, immediate value assignment, or arithmetic operation) before this API is called.
- The immediate address (reloads 1/2) is determined during the registration phase and is immutable at runtime. Use reloads 3/4 when a dynamic address is required at runtime.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: Load one variable from an immediate address (reload 1)
CcuResult MyKernel(CcuKernelArg arg) {
    Variable v;
    Load(0x10000000ULL, v);
    return CCU_SUCCESS;
}

// Scenario 2: Load four variables in batch from an immediate address (reload 2, num>1)
CcuResult MyKernel2(CcuKernelArg arg) {
    Array<Variable> vArr(4);    // Four physically contiguous variables
    Load(0x80000000ULL, vArr, 4);
    return CCU_SUCCESS;
}

// Scenario 3: Indirect load from a variable address (reload 3, the address is computed in the previous step)
CcuResult MyKernel3(CcuKernelArg arg) {
    Variable srcAddr, v;
    LoadArg(srcAddr, 0);    // Address passed in by the host
    Load(srcAddr, v);       // Runtime indirect addressing
    return CCU_SUCCESS;
}
```
