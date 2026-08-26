# EndpointLocType

<!-- md-trans-meta sourceCommit=7ff807bedd6173de4c7cb9ba16dadd5138b23868 translatedAt=2026-08-14T10:49:09.410Z pushedAt=2026-08-18T11:57:26.364Z -->

## Description

Defines the location of the communication device endpoint.

## Prototype

```c
typedef enum {
    ENDPOINT_LOC_TYPE_RESERVED = -1,  /* Reserved endpoint location */
    ENDPOINT_LOC_TYPE_DEVICE = 0,     /* Endpoint on the device */
    ENDPOINT_LOC_TYPE_HOST = 1,       /* Endpoint on the host */
} EndpointLocType;
```
