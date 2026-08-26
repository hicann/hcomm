# ThreadConfig

<!-- md-trans-meta sourceCommit=ab9c9ca78e7de405aea4bf444ae1790c36f65b04 translatedAt=2026-08-14T10:52:14.822Z pushedAt=2026-08-19T00:53:30.961Z -->

## Description

Thread configuration structure used to configure the number of synchronization resources per thread.

## Prototype

```c
typedef struct {
    CommAbiHeader header;              /* ABI header containing version and other information */
    uint16_t notifyNumPerThread;       /* Number of synchronization resources (Notify) in each communication thread */
    uint8_t reserved[14];              /* Reserved field */
} ThreadConfig;
```

## Member Description

| Member | Description |
| --- | --- |
| header | ABI header containing version and other information. For the definition of the CommAbiHeader type, see [CommAbiHeader](CommAbiHeader.md). |
| notifyNumPerThread | Number of synchronization resources (Notify) in each communication thread. The value ranges from 0 to 65535. Limited by the underlying Notify resource pool, the total number of synchronization resources of all threads in a communicator cannot exceed 65536. |
| reserved | Reserved field. |
