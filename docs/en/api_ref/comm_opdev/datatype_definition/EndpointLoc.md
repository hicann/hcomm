# EndpointLoc

<!-- md-trans-meta sourceCommit=7ff807bedd6173de4c7cb9ba16dadd5138b23868 translatedAt=2026-08-14T10:49:07.005Z pushedAt=2026-08-18T11:57:10.292Z -->

## Description

Structure defining the endpoint location type.

## Prototype

```c
typedef struct {
    EndpointLocType locType;        /* Location type of the endpoint */
    union {
        uint8_t raws[60];           /* Generic data */
        struct {
            uint32_t devPhyId;      /* Physical ID of the device */
            uint32_t superDevId;    /* SuperPoD device ID */
            uint32_t serverIdx;     /* Server index */
            uint32_t superPodIdx;   /* SuperPoD position index */
        } device;                   /* Used when locType is DEVICE. */
        struct {
            uint32_t id;            /* Common ID, used when locType is HOST or others. */
        } host;
    };
} EndpointLoc;
```
