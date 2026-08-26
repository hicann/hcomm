# Loop

<!-- md-trans-meta sourceCommit=97c142fbd6f7bfa37c6fcae34433680b079af61d translatedAt=2026-08-14T10:00:01.722Z pushedAt=2026-08-18T01:51:18.875Z -->

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

Hands the loop body to the CCU loop execution engine for automatic iteration. It is a hardware loop class in the CCU kernel. When a `ccu::Loop` object is constructed, the body is recorded immediately (the body lambda is executed once during construction). At runtime, the hardware automatically advances the iteration, so there is no need to manually write address increment or counter update logic in the body.

Compared with a software loop ([CCU_WHILE](CCU_WHILE.md)), a hardware loop does not re-execute the body instructions in each iteration. Instead, the loop execution engine automatically advances the address/buffer/event by offset. It is suitable for scenarios with a large number of iterations of the same structure where the body contains only local data movement operations, with minimal instruction overhead.

> [!CAUTION] Caution
> Constructing `ccu::Loop` alone only records the body and does not actually deliver the hardware loop instruction. You must add `Loop` to [`ccu::LoopGroup`](LoopGroup.md). During the registration phase, `LoopGroup` writes parameters such as `iterNum / addrOffset` and synthesizes the hardware loop instruction, after which the hardware actually iterates according to the configuration. If you write `Loop l(cfg, body);` directly without adding it to any LoopGroup, the body is executed only once (during recording at registration), and no loop effect is produced at runtime.

## Class Definition

```cpp
namespace AscendC {
namespace ccu {

class Loop {
public:
    // Construction method 1: config-based (iteration parameters are known during registration)
    Loop(const CcuLoopConfig &loopCfg, const Func &func);

    // Construction method 2: var-based (iteration parameters are determined by variable at runtime)
    Loop(Variable &loopCfg, const Func &func);
};

} // namespace ccu
} // namespace AscendC
```

## Parameters

### Construction Method 1: config-based

| Parameter | Input/Output | Description |
| --- | --- | --- |
| loopCfg | Input | Loop configuration. The type is `CcuLoopConfig`. For the meaning of each field, see the following table. The iteration count and address offset are determined during kernel registration. |
| func | Input | Loop body. The type is `ccu::Func`, which must be a `Func` with no input parameters (a lambda with no parameters). During construction, the body lambda is executed once immediately. |

### Construction Method 2: var-based

| Parameter | Input/Output | Description |
| --- | --- | --- |
| loopCfg | Input | Loop configuration. The type is `ccu::Variable`. Parameters such as the iteration count are determined by the variable value at runtime. This parameter contains 64 bits, where [12:0] indicates the iteration count of the Loop, [44:13] indicates the address offset used by data transfer instructions in each iteration of the loop, data address = value of the `Address` object + [44:13] * iteration count, and [52:45] indicates the EngineID used by the loop. The user must fill in 0, and the runtime framework fills in this value. |
| func | Input | Same as construction method 1. |

### CcuLoopConfig

It is the parameter structure used by config-based construction. The fields are as follows:

| Field | Type | Description |
| --- | --- | --- |
| `addrOffset` | `uint64_t` | Number of bytes by which `Address` in the body is automatically offset in each iteration. |
| `iterNum` | `uint64_t` | Total number of iterations. |

## Exceptions

When `ccu::Loop` fails to be constructed, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code). Common causes:

| Cause | Error Code |
| --- | --- |
| The `func` lambda has input parameters. (The loop requires a lambda with no parameters.) | `CCU_E_PARA` |
| Insufficient hardware resources, etc. | `CCU_E_UNAVAIL`, etc. |

## Constraints

- During construction of `ccu::Loop`, the body is recorded immediately. After construction is complete, no more content can be appended to the loop.
- The `Func` lambda of the body must have no input parameters. If it has input parameters, an exception is thrown during construction.
- The loop body has the following restrictions:
  - Will be rejected (returns `CCU_E_NOT_SUPPORT`): Neither of the two reloads of [EventRecord](../synchronization/EventRecord.md), namely `EventRecord(Event)` and `EventRecord(const char*)` (that is, [LocalNotifyRecord](../synchronization/LocalNotifyRecord.md)), is available in the body.
  - Will throw an exception (error code `CCU_E_INTERNAL`): calling [CallFunc](CallFunc.md) (`ccu::CallFunc<F>`) in the body.
  - Nesting software control flow macros such as [CCU_IF](CCU_IF.md), [CCU_WHILE](CCU_WHILE.md), and [CCU_DO](CCU_DO.md) in the body is not allowed.
  - It is recommended to avoid calling [NotifyRecord](../synchronization/NotifyRecord.md) and [WriteVariableWithNotify](../synchronization/WriteVariableWithNotify.md) in the body. These two APIs do not report errors in the body, but the loop body is expanded in parallel, and the remote notify semantics are not unique in a parallel environment. Use them with caution.
- `ccu::Loop` must be added to `ccu::LoopGroup` before a hardware loop instruction can be delivered (for details, see [Description](#description)). Do not use it alone.
- A `ccu::Loop` object cannot be added to a `ccu::LoopGroup` repeatedly.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Move four consecutive 4 KB data blocks from on-chip memory to CcuBuffer in sequence.
// The hardware loop automatically advances the address by addrOffset, so there is no need to manually increment the address.
CcuResult MyKernel(CcuKernelArg arg) {
    Variable r1, numA, numB;
    numA = 10;
    numB = 20;

    // Define the body: a lambda with no input parameters.
    Func body([&] {
        r1 = numA + numB;   // Perform addition in each iteration.
    });

    // config-based: fixed 4 iterations, with the address offset by 4096 bytes in each iteration.
    CcuLoopConfig cfg;
    cfg.addrOffset = 4096;
    cfg.iterNum = 4;
    Loop l(cfg, body);   // During construction, the body is recorded immediately (but only the IR is recorded; no hardware loop instruction is delivered).

    // Key point: The loop instruction is actually delivered only after the loop is added to a LoopGroup.
    CcuLoopGroupConfig grpCfg{};   // For a single-loop scenario, set all fields to 0.
    LoopGroup g(grpCfg, /*maxLoopNum=*/1, {l});

    return CCU_SUCCESS;
}
```
