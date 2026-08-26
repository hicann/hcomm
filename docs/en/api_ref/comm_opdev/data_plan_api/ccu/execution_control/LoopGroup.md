# LoopGroup

<!-- md-trans-meta sourceCommit=ad20ba1b62c0a7bfacd6aa282d8825ba68d368fc translatedAt=2026-08-14T10:00:26.591Z pushedAt=2026-08-18T02:09:21.025Z -->

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

Organizes multiple `ccu::Loop` objects into a group to share the same loop execution engine resource pool, preventing resource exhaustion caused by multiple independent `Loop` objects each exclusively occupying loop execution engine resources. It is a hardware LoopGroup class in the CCU kernel.

When constructing `ccu::LoopGroup`, each `Loop` in the passed `loops` list is automatically added to the group.

## Class Definition

```cpp
namespace AscendC {
namespace ccu {

class LoopGroup {
public:
    // Construction method 1: config-based (Group parameters are known at registration time.)
    LoopGroup(const CcuLoopGroupConfig &loopGroupCfg, uint32_t maxLoopNum,
              const std::vector<Loop> &loops);

    // Construction method 2: var-based (Group parameters are determined by variable at runtime.)
    LoopGroup(Variable &parallelCfg, Variable &offsetCfg, uint32_t maxLoopNum,
              const std::vector<Loop> &loops);
};

} // namespace ccu
} // namespace AscendC
```

## Parameters

### Construction Method 1: config-based

| Parameter | Input/Output | Description |
| --- | --- | --- |
| loopGroupCfg | Input | LoopGroup configuration, of the `CcuLoopGroupConfig` type. For details about its fields, see the following table. |
| maxLoopNum | Input | Maximum number of loops that this group can hold. The framework reserves the loop execution engine resource pool capacity accordingly. |
| loops | Input | List of `ccu::Loop` objects to be added to this group. Each loop in the list is automatically registered to the group in the `LoopGroup` constructor. |

### Construction Method 2: var-based

| Parameter | Input/Output | Description |
| --- | --- | --- |
| parallelCfg | Input | Parallel configuration variable, which determines the parallel parameters at runtime. This parameter contains 64 bits, where [47:41] indicates the number of loop instructions contained in the loop group, [54:48] indicates the loop offset at which the loop instructions contained in the loop group need to complete loop automatic unrolling, and [61:55] indicates the number of times the loop needs to be unrolled. Example: parallelCfg[47:41]=4 indicates that the program contains 4 loop instructions, parallelCfg[54:48]=1 indicates that unrolling starts from the loop numbered 1, and parallelCfg[61:55]=3 indicates that loop 1, loop 2, and loop 3 are each replicated and unrolled 3 times, while loop 0 is not replicated. After unrolling, the total number of loops is 4 + (4-1) * 3 = 13. |
| offsetCfg | Input | Offset configuration variable, which determines the offset parameters at runtime. This parameter contains 64 bits, where [9:0] indicates the event resource offset used after the loop is unrolled, [20:10] indicates the CcuBuffer resource offset used after the loop is unrolled, and [52:21] indicates the address accumulation offset used by each data transfer instruction after the loop is unrolled. |
| maxLoopNum | Input | Same as construction method 1. |
| loops | Input | Same as construction method 1. |

### CcuLoopGroupConfig

Parameter structure used by the config-based construction method. The fields are as follows:

| Field | Type | Description |
| --- | --- | --- |
| `cloneNum` | `uint32_t` | Number of parallel clones, which specifies the number of instances executed concurrently in the group. |
| `cloneLoopOffset` | `uint32_t` | Loop offset between clone instances. |
| `addrOffset` | `uint32_t` | Number of bytes by which the `Address` is offset between clone instances. |
| `ccuBufferOffset` | `uint32_t` | Number of slices by which the `CcuBuffer` slice is offset between clone instances. |
| `eventOffset` | `uint32_t` | Number of bits by which the `Event` slot is offset between clone instances. |

## Exceptions

When `ccu::LoopGroup` fails to be constructed, an exception is thrown (carrying a [CcuResult](../../../datatype_definition/CcuResult.md) error code). Common causes:

| Cause | Error Code |
| --- | --- |
| `maxLoopNum` is 0, or the runtime configuration variable of the var-based construction is empty. | `CCU_E_PARA` |
| The number of loops actually added exceeds `maxLoopNum` (insufficient loop execution engine resource pool capacity.) | `CCU_E_PARA` |
| Physical resources (`Variable`/`Address`/`Event`/`CcuBuffer`, etc.) are insufficient. | `CCU_E_UNAVAIL`, etc. |

## Constraints

- When `ccu::LoopGroup` is constructed, it immediately traverses the `loops` list and registers each loop to the group. After registration, the group members cannot be modified.
- The `ccu::Loop` objects in the `loops` list must have been constructed (that is, the body has been recorded) before the `ccu::LoopGroup` is constructed.
- The same `ccu::Loop` object should not be added to multiple `ccu::LoopGroup` objects.
- The body constraints of each loop in a group are the same as those of an independent `ccu::Loop` (see the constraints in [Loop](Loop.md)).
- `maxLoopNum` must be greater than 0. If it is 0, construction fails directly (`CCU_E_PARA`).
- `maxLoopNum` should be greater than or equal to the actual size of the `loops` list (that is, the number of loops that will actually be added to the group, including those reused by unrolling). If it is too small, adding loops later will fail due to insufficient loop execution engine resource pool capacity (`CCU_E_PARA`).

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Two Loops share the loop execution engine resource pool.
CcuResult MyKernel(CcuKernelArg arg) {
    Variable r1, r2, numA, numB;
    numA = 10; numB = 20;

    Func body1([&] { r1 = numA + numB; });
    Func body2([&] { r2 = numA + numA; });

    CcuLoopConfig cfg1;
    cfg1.addrOffset = 0;
    cfg1.iterNum = 2;
    Loop l1(cfg1, body1);

    CcuLoopConfig cfg2;
    cfg2.addrOffset = 0;
    cfg2.iterNum = 3;
    Loop l2(cfg2, body2);

    // Organize l1 and l2 into a LoopGroup to share the loop execution engine resource pool.
    CcuLoopGroupConfig grpCfg;
    grpCfg.cloneNum = 0;
    grpCfg.cloneLoopOffset = 0;
    grpCfg.addrOffset = 0;
    grpCfg.ccuBufferOffset = 0;
    grpCfg.eventOffset = 0;
    LoopGroup g(grpCfg, /*maxLoopNum=*/2, {l1, l2});

    return CCU_SUCCESS;
}
```
