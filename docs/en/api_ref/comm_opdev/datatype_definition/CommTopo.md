# CommTopo

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T10:47:33.736Z pushedAt=2026-08-18T11:54:14.060Z -->

## Description

Defines the communication topology types.

## Prototype

```c
typedef enum {
    COMM_TOPO_RESERVED = -1,  /* Reserved topology */
    COMM_TOPO_CLOS = 0,       /* CLOS interconnection topology */
    COMM_TOPO_1DMESH = 1,     /* 1DMesh interconnection topology */
    COMM_TOPO_910_93 = 2,     /* Interconnection topology of Atlas A3 training products/Atlas A3 inference products (with SIO) */
    COMM_TOPO_310P = 3,       /* Interconnection topology of Atlas inference products */
    COMM_TOPO_A2AXSERVER = 4, /* A2_AX_SERVER */
    COMM_TOPO_CUSTOM = 5      /* Custom */
} CommTopo;
```
