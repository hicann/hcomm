# HcommChannelDesc

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T10:50:36.427Z pushedAt=2026-08-19T00:45:21.304Z -->

## Description

Defines inter-component channel parameters.

## Prototype

```c
typedef struct {
    CommAbiHeader header;             /* ABI header, containing version and other information */
    EndpointDesc remoteEndpoint;      /* Remote network device endpoint description */
    uint32_t notifyNum;               /* Number of synchronization signals used on the channel */

    // When exchangeAllMems = True, memHandle does not need to be configured.
    bool exchangeAllMems;             /* Indicates whether to exchange the memory information registered on the local network device endpoint. */
    HcommMemHandle *memHandles;       /* Memory handles registered to the communicator to be exchanged. Invalid when exchangeAllMems is true. */
    uint32_t memHandleNum;            /* Number of memory handles registered to the communicator to be exchanged. Invalid when exchangeAllMems is true. */
    HcommSocket socket;               /* Socket handle. */
    HcommSocketRole role;             /* Local role (SERVER or CLIENT). */
    uint16_t port;                    /* Socket listens on the specified port */
    union {
        uint8_t raws[128];            /* General cache */
        struct {
            uint32_t queueNum;        /* Number of QPs. Currently, only one QP is supported. */
            uint32_t retryCnt;        /* Maximum number of retransmissions, ranging from 0 to 7 and defaulting to 7 */
            uint32_t retryInterval;   /* Retransmission interval, ranging from 5 to 24 and defaulting to 20 (corresponding to 4.096*2^20 us) */
            uint8_t tc;               /* Traffic class (QoS), ranging from 0 to 255 and defaulting to 132 */
            uint8_t sl;               /* Service level (QoS), ranging from 0 to 7 and defaulting to 4 */
            uint32_t qpThreshold;     /* Minimum data size per QP in multi-QP scenarios (B) */
        } roceAttr;
        struct {
            uint32_t qos;             /* HCCS QoS */
        } hccsAttr;
        struct {
            uint32_t sqDepth;         /* UB queue depth. 0 indicates using the default value. */
        } ubAttr;
    };
    uint32_t qos;             /* Decouple the communicator QoS from the protocol. */
} HcommChannelDesc;
```
