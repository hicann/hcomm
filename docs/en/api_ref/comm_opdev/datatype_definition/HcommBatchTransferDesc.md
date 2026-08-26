# HcommBatchTransferDesc

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-14T10:50:17.044Z pushedAt=2026-08-19T00:51:21.130Z -->

## Description

Batch transfer descriptor structure, used to describe the parameters of a single transfer task. The transfer type is specified by transType, and the detailed parameters of the corresponding type are provided by the transferInfo union.

## Prototype

```c
typedef struct {
    HcommTransferType transType;    /* Transfer type. See HcommTransferType. */
    uint8_t reserved[4];            /* Reserved fields */
    union {
        uint8_t raws[56];           /* Raw byte access, used for scenarios such as zero-copy serialization. */
        struct {
            uint64_t len;           /* Data length (in bytes) */
            void *dst;              /* Remote destination address */
            void *src;              /* Local source address */
        } write;
        struct {
            uint64_t len;           /* Data length (in bytes) */
            void *dst;              /* Local destination address */
            void *src;              /* Remote source address */
        } read;
        struct {
            uint64_t count;         /* Number of elements */
            void *dst;              /* Remote destination address */
            void *src;              /* Local source address */
            HcommReduceOp reduceOp; /* Reduction operation type */
            HcommDataType dataType; /* Data type */
        } reduce;
        struct {
            uint32_t notifyIdx;     /* Notification index */
        } notifyRecord;
        struct {
            uint64_t len;           /* Data length (in bytes) */
            void *dst;              /* Remote destination address */
            void *src;              /* Local source address */
            uint32_t notifyIdx;     /* Remote notification index for notifying after the write completes. */
        } writeWithNotify;
        struct {
            uint64_t count;         /* Number of elements. */
            void *dst;              /* Remote destination address. */
            void *src;              /* Local source address. */
            HcommReduceOp reduceOp; /* Reduction operation type. */
            HcommDataType dataType; /* Data type */
            uint32_t notifyIdx;     /* Remote notification index for notification after write reduction is complete */
        } writeReduceWithNotify;
    } transferInfo;
} HcommBatchTransferDesc;
```
