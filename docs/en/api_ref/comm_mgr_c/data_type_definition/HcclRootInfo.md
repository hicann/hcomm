# HcclRootInfo

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T08:26:20.952Z pushedAt=2026-08-14T08:49:42.842Z -->

## Description

Rank information of the root node, mainly including the host IP address and host port of the root node, as well as the unique identifier of the root node (obtained by concatenating information such as the device ID and timestamp).

## Prototype

```c
const uint32_t HCCL_ROOT_INFO_BYTES =  4108; // 4108: root info length
typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;
```
