# Introduction

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T10:15:52.904Z pushedAt=2026-08-18T07:17:27.445Z -->

This section provides the synchronization APIs in the CCU kernel for coordinating the completion order of asynchronous operations through Event and Notify.

In the CCU kernel, data movement and cross-kernel stage execution proceed asynchronously. When an explicit "ordering relationship" needs to be established between different kernels or different movement operations, the producer side calls the Record API to write a completion signal, and the consumer side calls the Wait API to block until the signal arrives. Based on the positional relationship between the producer and the consumer, synchronization is classified into the following three types:

| Type | Applicable Scenario | Producer-side API | Consumer-side API |
| --- | --- | --- | --- |
| Local event | Same kernel: waits for the completion of asynchronous movement initiated within this kernel. | [EventRecord](EventRecord.md) | [EventWait](EventWait.md) |
| Local Notify | Cross-kernel within the same die: coordinates the order between different kernels, paired by string tags. | [LocalNotifyRecord](LocalNotifyRecord.md) | [LocalNotifyWait](LocalNotifyWait.md) |
| Remote Notify | Cross-die (including cross-device and cross-node): transmits signals through channels. | [NotifyRecord](NotifyRecord.md) | [NotifyWait](NotifyWait.md) |

[WriteVariableWithNotify](WriteVariableWithNotify.md) is an extension API of remote Notify. It combines "writing a remote variable value" and "triggering a remote Notify" into an atomic operation, and is used in scenarios where a scalar value needs to be sent to the peer end together with a completion signal.

## API List

- [EventRecord](EventRecord.md)
- [EventWait](EventWait.md)
- [LocalNotifyRecord](LocalNotifyRecord.md)
- [LocalNotifyWait](LocalNotifyWait.md)
- [NotifyRecord](NotifyRecord.md)
- [NotifyWait](NotifyWait.md)
- [WriteVariableWithNotify](WriteVariableWithNotify.md)
