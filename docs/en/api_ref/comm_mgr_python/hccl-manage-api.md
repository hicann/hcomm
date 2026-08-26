# hccl.manage.api

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T09:04:16.950Z pushedAt=2026-08-15T08:52:47.193Z -->

The hccl.manage.api module provides collective communication group management APIs, including group creation and destruction and rank information query. All APIs in this module must be called after collective communication initialization is complete.

## API List

- [create_group](create_group.md): Creates a collective communication group named after the group.
- [destroy_group](destroy_group.md): Destroys a group.
- [get_rank_size](get_rank_size.md): Gets the number of ranks in a group.
- [get_rank_id](get_rank_id.md): Gets the rank ID corresponding to the device in a group.
- [get_local_rank_size](get_local_rank_size.md): Gets the number of local ranks on the server where the device in the group resides.
- [get_local_rank_id](get_local_rank_id.md): Gets the local rank ID on the server where the device in the group resides.
- [get_world_rank_from_group_rank](get_world_rank_from_group_rank.md): Gets the world rank ID based on the rank ID of the process in the group.
- [get_group_rank_from_world_rank](get_group_rank_from_world_rank.md): Gets the group rank ID of the process in the group based on the world rank ID.
