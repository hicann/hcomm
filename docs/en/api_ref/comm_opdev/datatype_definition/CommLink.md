# CommLink

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T10:46:28.599Z pushedAt=2026-08-18T11:51:12.012Z -->

## Description

Communication connection information, including the protocol, address, and other information required for creating a communication channel.

## Prototype

```c
typedef struct {
    CommAbiHeader header; /*  Compatible with Abi fields */
    EndpointDesc srcEndpointDesc; /*  Source endpoint description type */
    EndpointDesc dstEndpointDesc; /*  Destination endpoint description type */
    union {
        uint8_t raws[128];
        struct {
            CommProtocol linkProtocol; /* Communication protocol type */
            uint8_t hop; /* Link hop count */
        };
    } linkAttr;
} CommLink;
```
