# CommMemType

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T10:47:05.041Z pushedAt=2026-08-18T11:51:58.970Z -->

## Description

Memory physical location type.

## Prototype

```c
typedef enum {
    COMM_MEM_TYPE_INVALID = -1,   /* Invalid memory type */
    COMM_MEM_TYPE_DEVICE = 0,     /* Device-side memory (such as NPU) */
    COMM_MEM_TYPE_HOST,           /* Host-side memory */
} CommMemType;
```
