# CommAbiHeader

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T10:45:30.652Z pushedAt=2026-08-18T11:47:55.258Z -->

## Description

Structure compatible with Abi fields.

## Prototype

```c
typedef struct {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
} CommAbiHeader;
```
