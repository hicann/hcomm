# HcclKernelLaunchCfg

<!-- md-trans-meta sourceCommit=d37db64e2da4bd5cea0b7da10eef869022c9e3e4 translatedAt=2026-08-14T08:23:39.723Z pushedAt=2026-08-14T08:38:36.489Z -->

## Description

Describes the configuration information for running a kernel function on AI CPU, including configuration parameters such as the timeout period.

## Prototype

```c
typedef struct {
    CommAbiHeader header;
    uint64_t timeOut;
    uint8_t reserved[104];
} HcclKernelLaunchCfg;
```

## Parameters

- **header**: ABI header, which contains information such as the version. For the type definition, see [CommAbiHeader](../../comm_opdev/datatype_definition/CommAbiHeader.md).
- **timeOut**: Timeout period for the task scheduler to wait for task execution, in seconds.
- **reserved**: Reserved field, 104 bytes in length, for future extension.
