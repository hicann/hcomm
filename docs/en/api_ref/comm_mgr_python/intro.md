# Introduction

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-14T09:06:10.484Z pushedAt=2026-08-15T08:59:48.816Z -->

The HCCL Python APIs are used to implement framework adaptation in graph mode. Currently, they are only used for distributed optimization of TensorFlow networks on NPUs.

## Related Concepts

| Concept | Description |
| --- | --- |
| Group | A process group that participates in collective communication, including:<br>  - hccl_world_group: default global group that contains all ranks participating in collective communication. It is created through the rank table file.<br>  - Custom group: a subset of the process group contained in hccl_world_group. You can use the create_group API to define the ranks in the rank table as different groups to execute collective communication algorithms in parallel. |
| Rank | Each communication entity in a group is called a rank. Each rank is assigned a unique identifier ranging from 0 to n-1 (n is the number of NPUs). |
| Rank size | - Rank size: number of ranks in the entire group.<br>  - Local rank size: number of ranks of the processes in the group on the server where they reside. |
| Rank ID | - Rank ID: rank identifier of a process in a group. Value range: 0 to (rank size - 1). For a custom group, ranks are renumbered starting from 0 within the group. For hccl_world_group, the rank ID is the same as the world rank ID.<br>  - World rank ID: rank identifier of a process in hccl_world_group. Value range: 0 to (rank size - 1).<br>  - Local rank ID: rank identifier of a process in the group on the server where it resides. Value range: 0 to (local rank size - 1). |
