# get_rank_id

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T09:02:51.255Z pushedAt=2026-08-15T08:40:41.024Z -->

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

Obtains the rank ID of a device in a group.

## Function Prototype

```python
def get_rank_id(group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name. If this parameter is not configured, the default value "hccl_world_group" is used. |

## Return Value

int type. Returns the rank ID of the device in the group.

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the input parameter group of this API. Calling this API from a rank outside this range will fail.
- After [create_group](create_group.md) is complete, call this API to obtain the rank ID of the process in the group.
- If "hccl_world_group" is passed in, the rank ID of the process in hccl_world_group is returned.

## Example

```python
from hccl.manage.api import create_group
from hccl.manage.api import get_rank_id
create_group("myGroup", 4, [0, 1, 2, 3])
rankId = get_rank_id("myGroup")
```
