# get_group_rank_from_world_rank

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:00:21.298Z pushedAt=2026-08-15T08:22:45.925Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

Obtains the group rank ID of a process in a group based on its world rank ID.

## Function Prototype

```python
def get_group_rank_from_world_rank(world_rank_id, group)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| world_rank_id | Input | int type.<br>Rank ID of the process in "hccl_world_group". |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name, which can be a user-defined group or "hccl_world_group". |

## Return Value

int type. The rank ID of the process in the group is returned on success.

## Constraints

- This API can be called only after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of this API. Calling this API from a rank outside this range will fail.
- After [create_group](create_group.md) is complete, call this API to convert a world rank ID to a group rank ID.
- This API and [get_world_rank_from_group_rank](get_world_rank_from_group_rank.md) are inverse operations of each other. Note the difference in parameter order: this API uses the parameter order (world_rank_id, group), while its inverse operation uses (group, group_rank_id).

## Example

```python
from hccl.manage.api import create_group
from hccl.manage.api import get_group_rank_from_world_rank
create_group("myGroup", 4, [0, 1, 2, 3])
groupRankId = get_group_rank_from_world_rank(1, "myGroup")
```
