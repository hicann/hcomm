# HcclChannelDesc

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T10:49:37.020Z pushedAt=2026-08-19T00:37:55.940Z -->

## Description

Defines communication channel parameters.

## Prototype

```c
typedef struct {
    CommAbiHeader header;
    uint32_t remoteRank;              /* Remote rank ID */
    CommProtocol channelProtocol;     /* Communication protocol */
    EndpointDesc localEndpoint;       /* Local network device endpoint description, only supported on Ascend 950PR/Ascend 950DT */
    EndpointDesc remoteEndpoint;      /* Description of the remote network device endpoint. Only supported on Ascend 950PR/Ascend 950DT. */
    uint32_t notifyNum;               /* Number of synchronization signals used on the channel. The value ranges from 0 to 64, and the default value is 0. */
    HcclMemHandle *memHandles;        /* Memory handle to be exchanged that is registered to the communicator. Only supported on the AIV engine of Ascend 950PR/Ascend 950DT. */
    uint32_t memHandleNum;            /* Number of memory handles to be exchanged that are registered to the communicator. Only supported on the AIV engine of Ascend 950PR/Ascend 950DT. */
    union {
        uint8_t raws[128];            /* General cache. */
        struct {
            uint32_t queueNum;        /* Number of QPs. Currently, only one QP is supported. */
            uint32_t retryCnt;        /* Maximum number of retransmissions, ranging from 0 to 7. The default value is 7. */
            uint32_t retryInterval;   /* Retransmission interval, ranging from 5 to 24. The default value is 20 (corresponding to 4.096*2^20 us). */
            uint8_t tc;               /* Traffic class (QoS), ranging from 0 to 255. The default value is 132. */
            uint8_t sl;               /* Service level (QoS), ranging from 0 to 7. The default value is 4. */
        } roceAttr;
    };
} HcclChannelDesc;
```
