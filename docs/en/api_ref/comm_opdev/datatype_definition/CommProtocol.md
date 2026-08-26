# CommProtocol

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T10:47:21.087Z pushedAt=2026-08-18T11:53:33.612Z -->

## Description

Defines the communication protocol type enumeration.

## Prototype

```c
typedef enum {
    COMM_PROTOCOL_RESERVED = -1,  /* Reserved protocol type */
    COMM_PROTOCOL_HCCS = 0,       /* HCCS protocol */
    COMM_PROTOCOL_ROCE = 1,       /* RDMA over Converged Ethernet */
    COMM_PROTOCOL_PCIE = 2,       /* PCIE protocol */
    COMM_PROTOCOL_SIO = 3,        /* SIO protocol */
    COMM_PROTOCOL_UBC_CTP = 4,    /* Huawei Unified Bus UBC_CTP */
    COMM_PROTOCOL_UBC_TP = 5,     /* Huawei Unified Bus UBC_TP */
    COMM_PROTOCOL_UB_MEM = 6,     /* UB_MEM */
    COMM_PROTOCOL_UBOE = 7,       /* UBoE */
    COMM_PROTOCOL_HCCS_ONLY = 8,  /* HCCS is used for two dies on one device */
} CommProtocol;
```

## Supported Products

For Ascend 950PR/Ascend 950DT, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_CPU
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_UBOE
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    - COMM_PROTOCOL_UB_MEM
    - COMM_PROTOCOL_UBG
  - COMM_ENGINE_AIV
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    - COMM_PROTOCOL_UB_MEM
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_CCU
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP

For Atlas A3 training products and Atlas A3 inference products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY

For Atlas A2 training products and Atlas A2 inference products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_CPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY
