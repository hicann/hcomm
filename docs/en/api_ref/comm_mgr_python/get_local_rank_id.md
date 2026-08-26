# get_local_rank_id

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T09:00:44.234Z pushedAt=2026-08-15T08:35:50.230Z -->

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

Obtains the local rank ID of a device in a group.

## Function Prototype

```python
def get_local_rank_id(group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name. If this parameter is not configured, the default value "hccl_world_group" is used. |

## Return Value

int type. Returns the local rank ID of the device on the server where it resides.

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. If a rank outside this range calls this API, the call fails.
- After [create_group](create_group.md) is complete, call this API to obtain the local rank ID of the process in the group.
- If "hccl_world_group" is passed in, the local rank ID of the process in world_group is returned.

## Example

```python
from hccl.manage.api import create_group
from hccl.manage.api import get_local_rank_id
create_group("myGroup", 4, [0, 1, 2, 3])
localRankId = get_local_rank_id("myGroup")
```
