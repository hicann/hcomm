# ThreadResType

<!-- md-trans-meta sourceCommit=5d2b6a1c1d3ecb2aedf23e593d18044c239c2ddc translatedAt=2026-08-14T10:52:51.995Z pushedAt=2026-08-19T00:54:25.222Z -->

## Description

Underlying resource types that can be obtained.

## Prototype

```c
typedef enum {
    THREAD_RES_TYPE_INVALID = -1,
    THREAD_RES_TYPE_STREAM = 0,   // The obtained resource type is stream.
} ThreadResType;
```

## Field Description

| Field | Value | Description |
| --- | --- | --- |
| THREAD_RES_TYPE_INVALID | -1 | Invalid resource type. |
| THREAD_RES_TYPE_STREAM | 0 | Stream resource. The corresponding resource type is [ThreadResTypeStream](ThreadResTypeStream.md), which can be obtained through the [HcclThreadResGetInfo](../control_plane_api/comms_domain_resource_mgmt/HcclThreadResGetInfo.md) API. |
