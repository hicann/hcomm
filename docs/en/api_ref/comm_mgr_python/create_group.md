# create_group

<!-- md-trans-meta sourceCommit=298b07499d8a20c7091e3ae605c9b7392c961870 translatedAt=2026-08-14T08:59:33.324Z pushedAt=2026-08-15T08:33:19.721Z -->

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

Creates a user-defined group for collective communication.

If developers do not call this API to create a user-defined group, all devices participating in cluster training are grouped into the global hccl_world_group by default.

A group is a process group participating in collective communication, where:

- hccl_world_group: default global group that contains all ranks participating in collective communication and is automatically created by HCCL.
- Custom group: a subset of the process group contained in hccl_world_group.

## Function Prototype

```python
def create_group(group, rank_num, rank_ids)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name, which is the identifier of the collective communication group. It cannot be the default global group name "hccl_world_group". If the group name passed in by the user is "hccl_world_group", the creation fails. |
| rank_num | Input | int type.<br>Number of ranks that make up the group.<br>The maximum value is 32768. |
| rank_ids | Input | list type.<br>List of world_rank_id values that make up the group. |

## Return Value

None

## Constraints

- It must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. A rank outside this range fails to call this API.
- The rank_ids parameter has different restrictions on different board types.

  **For Atlas A3 training products/Atlas A3 inference products**: It is recommended that the number of servers in each SuperPoD be the same and the number of ranks in each server be the same. Otherwise, performance degradation may occur.

  **For Atlas A2 training products/Atlas A2 inference products:**
  - In the single-server scenario, there are no restrictions on rank_ids.
  - In the server cluster scenario, rank_ids must meet the following conditions:

    It is recommended that each server select the same number of ranks (any number is acceptable), and the corresponding positions of the ranks selected by each server must be equal (that is, the rank IDs are equal modulo 8). If each server selects a different number of ranks, performance degradation will occur.

    Example:

    Assume that a group is created for three servers, and the rank IDs of the three servers are as follows:

    {0,1,2,3,4,5,6,7}

    {8,9,10,11,12,13,14,15}

    {16,17,18,19,20,21,22,23}

    Then the rank_ids lists that meet the requirement can be:

    rank_ids=[1,9,17]

    rank_ids=[1,2,9,10,17,18]

    rank_ids=[4,5,6,7,12,13,14,15,20,21,22,23]

  <!-- npu="910" id6 -->

  **For Atlas training products:**
  - In the single-server scenario, rank_ids must meet the following conditions:

    The number of ranks must be 1/2/4/8. Devices 0-3 and devices 4-7 each form a network. When the number of ranks is 2/4, the selected AI processors must belong to the same cluster.
  - In the server cluster scenario, rank_ids must meet the following conditions:
    - Each server must select the same number of ranks (and the number must be 1/2/4/8).
    - When the number of ranks selected by each server is 2 or 4, the selected AI processors must belong to the same cluster (that is, the remainders of the rank IDs modulo 8 are all less than 4 or all greater than or equal to 4).

       Example:

       Assume that a group is created for three servers, and the rank IDs of the three servers are as follows:

       {0,1,2,3,4,5,6,7}

       {8,9,10,11,12,13,14,15}

       {16,17,18,19,20,21,22,23}

       Then the rank_ids lists that meet the requirements can be:

       rank_ids=[1,9,17]

       rank_ids=[1,2,9,10,17,18]

       rank_ids=[4,5,6,7,12,13,14,15,20,21,22,23]    
  <!-- end id6 -->
  <!-- npu="310p" id7 -->   
  **For Atlas 300I Duo inference devices**: Only the single-server scenario is supported, and there are no restrictions on rank_ids.
  <!-- end id7 -->

  Note: It is recommended that rank_ids be sorted according to the physical connection order of devices, that is, devices that are physically closer should be grouped together. For example, if device_ip is set in ascending order of physical connections, rank_ids should also be set in ascending order.

## Example

```python
from hccl.manage.api import create_group
create_group("myGroup", 4, [0, 1, 2, 3])
```
