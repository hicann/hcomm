# HcommSocketRole

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T10:51:46.409Z pushedAt=2026-08-19T00:47:22.151Z -->

## Description

Defines the socket role type.

## Prototype

```c
typedef enum {
    HCOMM_SOCKET_ROLE_RESERVED = -1, /* Reserved socket role */
    HCOMM_SOCKET_ROLE_CLIENT = 0,    /* Client role, used to initiate connections */
    HCOMM_SOCKET_ROLE_SERVER = 1,    /* Server role, used to listen for connections */
} HcommSocketRole;
```
