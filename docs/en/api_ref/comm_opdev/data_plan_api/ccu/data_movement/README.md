# Introduction

<!-- md-trans-meta sourceCommit=5a8b6510ac6a3236129bf053dfba6e7b50e61bdc translatedAt=2026-08-14T09:50:09.468Z pushedAt=2026-08-17T11:01:04.873Z -->

This section provides data movement APIs for moving bytes within the CCU kernel between local on-chip memory, local CcuBuffer, and cross-rank remote on-chip memory, with an optional reduction operation performed during the movement.

All data movement APIs are asynchronous. When the hardware completes the movement, it automatically sets bit `mask` of `event` to 1, and the downstream waits for the completion signal through `EventWait`. Based on the data path, the APIs are classified into the following two types:

| Type | Scenario | API |
| --- | --- | --- |
| Local operation | Copy and reduction between local on-chip memory and local on-chip memory, or between local on-chip memory and local CcuBuffer | [LocalCopy](LocalCopy.md), [LocalReduce](LocalReduce.md) |
| Cross-rank operation | Read and write data between local and remote on-chip memory through a channel, with optional reduction. | [Read](Read.md), [ReadReduce](ReadReduce.md), [Write](Write.md), [WriteReduce](WriteReduce.md) |

Cross-rank operations require that all `ChannelHandle` objects belong to the same die and are uniformly verified by the framework.

## API List

- [LocalCopy](LocalCopy.md)
- [LocalReduce](LocalReduce.md)
- [Read](Read.md)
- [ReadReduce](ReadReduce.md)
- [Write](Write.md)
- [WriteReduce](WriteReduce.md)
