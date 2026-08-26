# Selecting Algorithms

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-11T07:13:29.046Z pushedAt=2026-08-20T11:39:14.569Z -->

## Selection Strategy

Using the CCU engine can save on-chip memory bandwidth and reduce communication unrolling time compared with the AI CPU mode. In addition, it does not occupy compute hardware (such as AIV). However, CCU hardware resources are limited, so it is recommended for use in critical scenarios with extreme performance requirements, such as TP and EP parallelism.

**Figure 1** Communication algorithm selection diagram
![](figures/ccu_algo_select.png "Communication algorithm selection diagram")

As shown in the preceding figure, the CCU communication operator supports multiple algorithm implementations. Developers can select the optimal communication algorithm based on topology information:

- Mesh algorithm implementation: suitable for scenarios where the intra-server physical topology is Mesh.
- NHR algorithm implementation: suitable for multi-server scenarios where one rank is selected from each server for communication.

> [!NOTE] Note
>
> 1. If a communication operator has only one algorithm implementation, you can skip the algorithm selection step described in this section.
> 2. The algorithms described in this chapter are developer-implemented algorithms. The algorithms configured through the HCCL\_ALGO environment variable are HCCL built-in algorithms. For details about HCCL built-in algorithms, see "References > Introduction to Collective Communication Algorithms" in *[HCCL User Guide](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/README.md)*.

## Sample Code

The following code snippet shows the algorithm selection logic based on the selection strategy described in [Selection Strategy](#selection-strategy):

```c
CommEngine engine;
if (engine == CommEngine::COMM_ENGINE_CCU) {
    algName = "CCUAllGatherMesh";  // Select the Mesh algorithm of the CCU engine.
} else {
    return HCCL_E_NOT_SUPPORT;
}
```
