# HcclSendRecvItem

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-11T08:21:29.422Z pushedAt=2026-08-11T09:24:21.400Z -->

## Description

Used for batch point-to-point communication operations to define the basic information of each communication task, including the task type, data address, data count, data type, and peer rank ID.

## Prototype

```c
typedef struct HcclSendRecvItemDef {
    HcclSendRecvType sendRecvType;    /* Indicates whether the current task type is sending or receiving. */
    void *buf;        /* Buffer address for data send/receive */
    uint64_t count;   /* Number of data elements for sending/receiving */
    HcclDataType dataType;   /* Type of sent/received data */
    uint32_t remoteRank;     /* Rank ID of the data receiving/sending end in the communicator */
} HcclSendRecvItem;
```
