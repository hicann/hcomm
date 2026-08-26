# Introduction

<!-- md-trans-meta sourceCommit=97c142fbd6f7bfa37c6fcae34433680b079af61d translatedAt=2026-08-14T10:02:02.527Z pushedAt=2026-08-18T02:16:38.743Z -->

This section provides the APIs for expressing dynamic control flow in a CCU kernel, which are divided into three subclasses:

| Subclass | Scenario | API |
| --- | --- | --- |
| Software branch/loop | Flexible control flow, nested structures, and any CCU API can be used in the body. | [CCU_IF](CCU_IF.md), [CCU_ELSE](CCU_ELSE.md), [CCU_WHILE](CCU_WHILE.md), [CCU_DO](CCU_DO.md) |
| Hardware loop | A large number of iterations with the same structure, where the body contains local data movement operations, pursuing minimal instruction overhead. | [Loop](Loop.md), [LoopGroup](LoopGroup.md) |
| FuncBlock | The same logic is called in multiple places within a kernel, saving SRAM. | [Func](Func.md), [CallFunc](CallFunc.md) |

The following are suggestions for selecting among the three subclasses:

- Complex control flow logic, nesting, and multiple CCU operations in the body → use software branch/loop macros.
- A large number of iterations with the same structure and only local data movement operations in the body → use hardware loop (`ccu::Loop`/`ccu::LoopGroup`).
- The same logic is called twice or more within a kernel → use FuncBlock (`ccu::Func`+`ccu::CallFunc`).

The three subclasses can be combined: nesting a hardware loop inside a software loop (`CCU_WHILE`/`CCU_IF`) is allowed. Do not use reverse nesting — software control flow (`CCU_IF`/`CCU_WHILE`/`CCU_DO`) should not be used inside a hardware loop body. The framework does not enforce validation of reverse nesting, but the behavior is undefined, so do not use it. Only `CallFunc` inside a hardware loop body throws an exception carrying `CCU_E_INTERNAL` (after being uniformly caught at the kernel registration entry, the registration API returns `CCU_E_INTERNAL`).

## API List

- [CCU_IF](CCU_IF.md)
- [CCU_ELSE](CCU_ELSE.md)
- [CCU_WHILE](CCU_WHILE.md)
- [CCU_DO](CCU_DO.md)
- [Loop](Loop.md)
- [LoopGroup](LoopGroup.md)
- [Func](Func.md)
- [CallFunc](CallFunc.md)
