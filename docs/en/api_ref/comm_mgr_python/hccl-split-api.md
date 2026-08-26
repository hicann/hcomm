# hccl.split.api

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:06:03.968Z pushedAt=2026-08-15T08:50:37.311Z -->

The hccl.split.api module provides APIs for setting the reverse gradient split strategy, which are used for performance tuning of allreduce fusion. All APIs in this module must be called after collective communication initialization is complete.

## API List

- [set_split_strategy_by_idx](set_split_strategy_by_idx.md): Sets the reverse gradient split strategy in a collective communication group based on the gradient index ID.
- [set_split_strategy_by_size](set_split_strategy_by_size.md): Sets the reverse gradient split strategy in a collective communication group based on the gradient data size percentage.
