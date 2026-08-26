# HcommTransferType

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T10:51:58.960Z pushedAt=2026-08-19T00:49:49.831Z -->

## Description

Defines the transfer type enumeration used in batch transfer operations. It is used for the transType field in the [HcommBatchTransferDesc](HcommBatchTransferDesc.md) structure to specify the operation type performed by a single transfer descriptor.

## Prototype

```c
typedef enum {
    HCOMM_TRANSFER_TYPE_INVALID = -1,                       /* Invalid transfer type */
    HCOMM_TRANSFER_TYPE_WRITE = 0,                          /* One-sided write, corresponding to transferInfo.write */
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE = 1,                   /* One-sided write reduction, corresponding to transferInfo.reduce */
    HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY = 2,              /* One-sided write with notification, corresponding to transferInfo.writeWithNotify */
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY = 3,       /* One-sided write reduction with notification, corresponding to transferInfo.writeReduceWithNotify */
    HCOMM_TRANSFER_TYPE_READ = 4,                           /* One-sided read, corresponding to transferInfo.read */
    HCOMM_TRANSFER_TYPE_READ_REDUCE = 5,                    /* One-sided read reduction, corresponding to transferInfo.reduce */
    HCOMM_TRANSFER_TYPE_NOTIFY_RECORD = 6                   /* Record notification event, corresponding to transferInfo.notifyRecord */
} HcommTransferType;
```
