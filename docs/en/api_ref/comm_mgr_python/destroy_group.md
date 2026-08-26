# destroy_group

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T08:59:34.125Z pushedAt=2026-08-15T08:20:14.378Z -->

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

Destroys a user-defined collective communication group.

## Function Prototype

```python
def destroy_group(group)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name, which is the identifier of the collective communication group. |

## Return Value

None

## Constraints

- It must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group input parameter of the current API. If a rank is outside this range, the API call fails.
- For groups with the same name, [destroy_group](destroy_group.md) and [create_group](create_group.md) must be used together, and destroy_group must be called after create_group is complete.
- If the group passed by the user is hccl_world_group (the default group), destroying the group fails.

## Example

```python
from hccl.manage.api import create_group
from hccl.manage.api import destroy_group
create_group("myGroup", 4, [0, 1, 2, 3])
destroy_group("myGroup")
```
