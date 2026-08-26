# Introduction

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T09:45:47.840Z pushedAt=2026-08-17T09:04:04.976Z -->

This section provides the APIs for passing scalar values between variables and external data sources in a CCU kernel. They are used to declare the source or destination of runtime data during the registration phase.

During the registration phase, a variable is only a scalar placeholder and does not carry a value. The APIs in this section are responsible for exchanging data with the outside before/after kernel execution. External sources fall into the following two categories:

| Source/Destination | When Is the Address Determined | API |
| --- | --- | --- |
| `taskArgs[]` (injected by the host at each launch) | At runtime launch | [LoadArg](LoadArg.md) |
| On-chip memory immediate address (a constant fixed at registration) | Registration phase | [Load](Load.md) (reload 1/2), [Store](Store.md) (reload 1/2) |
| On-chip memory indirect address (determined by a variable at runtime) | Runtime | [Load](Load.md) (reload 3/4), [Store](Store.md) (reload 3/4) |

Both `Load` and `Store` automatically select the immediate addressing or indirect addressing path based on the type of the first parameter (`uint64_t` or `Variable`). Each path internally supports two granularities: a single variable and a batch `Array<Variable>`.

## API List

- [LoadArg](LoadArg.md)
- [Load](Load.md)
- [Store](Store.md)
