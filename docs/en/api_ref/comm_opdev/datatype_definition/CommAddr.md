# CommAddr

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T10:46:00.236Z pushedAt=2026-08-18T11:48:49.979Z -->

## Description

Communication device address description structure.

## Prototype

```c
static const uint32_t COMM_ADDR_EID_LEN = 16U;
typedef struct {
    CommAddrType type;         /* communication address type */
    union {
        uint8_t raws[36];      /* generic data */
        struct in_addr addr;   /* IPv4 address structure */
        struct in6_addr addr6; /* IPv6 address structure */
        uint32_t id;           /* identifier */
        uint8_t eid[COMM_ADDR_EID_LEN];  /* EID address type */
    };
} CommAddr;
```
