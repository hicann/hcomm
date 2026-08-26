# HcommEndpointCheckFeature

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:11:12.101Z pushedAt=2026-08-17T02:33:05.448Z -->

## Supported Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not supported

## Description

Queries whether a specified network endpoint supports a certain feature, without actually creating the endpoint.

By passing in the endpoint description and the feature type, it queries whether the underlying hardware/network driver supports the feature, and returns the result through the value parameter. This API is mainly used for capability detection before creating an endpoint, so that the upper layer can select different communication strategies based on hardware capabilities.

Currently, only the NPU Direct RDMA Async (NDA) feature can be queried.

## Function Prototype

```c
HcommResult HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc *endpointDesc, bool *value);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| featureType | Input | Feature type to be queried.<br>For the definition of the HcommEndpointFeatureType type, see [HcommEndpointFeatureType](../../datatype_definition/HcommEndpointFeatureType.md). |
| endpointDesc | Input | Endpoint description, used to specify the endpoint attributes (protocol, address, location, etc.) to be queried. The feature can be queried without actually creating the endpoint.<br>For the definition of the EndpointDesc type, see [EndpointDesc](../../datatype_definition/EndpointDesc.md).<br>This parameter cannot be a null pointer. |
| value | Output | Result of feature support.<br>true indicates that the feature is supported, and false indicates that it is not supported.<br>This parameter cannot be a null pointer. |

## Return Value

HcommResult: The API returns 0 on success and a non-zero value on failure.

## Constraints

- You do not need to create an endpoint before calling this API. You only need to fill in the EndpointDesc description information.
- When querying the HCOMM_ENDPOINT_FEATURE_NDA feature, the protocol in EndpointDesc must be COMM_PROTOCOL_ROCE and the endpoint location type must be ENDPOINT_LOC_TYPE_HOST. Otherwise, false is returned.
- This API is mainly used in the capability detection phase and should not be called frequently during communication.

## Example

```c
// Fill in the endpoint description information.
EndpointDesc endpointDesc;
// Refer to HcommEndpointCreate to fill in EndpointDesc.
...

// Query whether the NDA feature is supported.
bool isNdaSupported = false;
HcommResult ret = HcommEndpointCheckFeature(HCOMM_ENDPOINT_FEATURE_NDA, &endpointDesc, &isNdaSupported);
if (ret != 0) {
    printf("Failed to check feature, ret = %d\n", ret);
    return ret;
}

if (isNdaSupported) {
    // Use the NDA feature for communication.
    // ...
} else {
    // Fall back to other communication methods.
    // ...
}
```
