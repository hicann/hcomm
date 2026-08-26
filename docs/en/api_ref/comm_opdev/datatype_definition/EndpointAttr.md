# EndpointAttr

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T10:47:31.204Z pushedAt=2026-08-18T11:54:36.995Z -->

## Description

Defines the attributes of an endpoint.

## Prototype

```c
typedef  enum {
    ENDPOINT_ATTR_INVALID = -1, /*Invalid attribute*/
    ENDPOINT_ATTR_BW_COEFF =  0, /* Bandwidth attribute*/
    ENDPOINT_ATTR_DIE_ID = 1,   /* Die ID */
    ENDPOINT_ATTR_LOCATION = 2, /* Location of the endpoint*/
} EndpointAttr;
```
