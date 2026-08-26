# EndpointDesc

<!-- md-trans-meta sourceCommit=7ff807bedd6173de4c7cb9ba16dadd5138b23868 translatedAt=2026-08-14T10:48:33.963Z pushedAt=2026-08-18T11:56:01.530Z -->

## Description

Structure defining the endpoint description type.

## Prototype

```c
typedef struct {
    CommProtocol protocol;  /* Communication protocol */
    CommAddr commAddr;      /* Communication address */
    EndpointLoc loc;        /* Location information of the endpoint */
    union {
        uint8_t raws[52];   /* Generic data */
    };
} EndpointDesc;
```
