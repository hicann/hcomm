# set_split_strategy_by_idx

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:07:28.877Z pushedAt=2026-08-17T01:23:24.009Z -->

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

Sets the backward gradient splitting strategy in a collective communication group based on gradient index IDs to implement allreduce fusion, for performance tuning of collective communication.

## Function Prototype

```python
def set_split_strategy_by_idx(idxList, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| idxList | Input | List type.<br>List of gradient index IDs.<br>  - The gradient index ID list must be a non-negative, ascending sequence.<br>  - The gradient index IDs must be set based on the total number of gradient parameters of the model. Index IDs start from 0. The maximum value can be obtained as follows: perform training without calling the gradient split API to set a gradient split strategy. In this case, the script uses the default gradient split method in [set_split_strategy_by_size](set_split_strategy_by_size.md) for training. After training is complete, search for the "segment result" keyword in the INFO-level host training logs to obtain the gradient split segments, for example: segment index list: [0,107] [108,159]. The largest number in this segment sequence (for example, 159) is the maximum index value of the total gradient parameters. Note: The complete training process may cause log overwriting. In this case, you can modify the LogAgentMaxFileNum configuration item in "/var/log/npu/conf/slog/slog.conf" to increase the number of log files retained on the host side. Alternatively, you can perform only one iteration of training.<br>  - Gradient splitting supports a maximum of 8 segments.<br>  - For example, if the model has 160 parameters that generate gradients in total and needs to be split into three segments [0,20], [21,100], and [101,159], set idxList=[20,100,159]. |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name. It can be "hccl_world_group" or a custom group. The default value is "hccl_world_group". |

## Return Value

None

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. If a rank outside this range calls this API, the call fails.
- If you do not call the gradient splitting API to set a splitting strategy, the default reverse gradient splitting strategy is used.

  Default splitting strategy: The gradient is split into two segments by data size. The first segment accounts for 96.54% of the data, and the second segment accounts for 3.46% (in some cases, the gradient may be split into a single segment).

## Example

```python
from hccl.split.api import *
set_split_strategy_by_idx([20, 100, 159], "group")
```
