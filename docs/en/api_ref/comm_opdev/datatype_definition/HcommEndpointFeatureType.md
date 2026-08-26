# HcommEndpointFeatureType

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-14T10:51:09.369Z pushedAt=2026-08-19T00:46:33.287Z -->

## Description

Defines the underlying feature types, used to specify the feature type to be queried in the [HcommEndpointCheckFeature](../control_plane_api/basic_resource_mgmt/HcommEndpointCheckFeature.md) API.

## Prototype

```c
typedef enum {
    HCOMM_ENDPOINT_FEATURE_INVALID = -1,       /* Invalid feature type. Configuration is not supported yet. */
    HCOMM_ENDPOINT_FEATURE_NDA = 0,            /* NPU Direct RDMA Async feature */
} HcommEndpointFeatureType;
```
