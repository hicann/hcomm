# CommMem

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T10:46:30.277Z pushedAt=2026-08-18T11:51:42.017Z -->

## Description

Structure describing memory segment metadata.

## Prototype

```c
typedef struct {
    CommMemType type; /* Memory physical location type */
    void *addr;       /* Memory address */
    uint64_t size;    /* Number of bytes in the memory region */
} CommMem;
```
