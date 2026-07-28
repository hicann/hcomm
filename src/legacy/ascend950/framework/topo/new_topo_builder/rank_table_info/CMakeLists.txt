if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        rank_table_info.cc
        rank_level_info.cc
        new_rank_info.cc
        changed_rank_info.cc
        control_plane.cc
        address_info.cc
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        rank_level_info.cc
        rank_table_info.cc
        new_rank_info.cc
        control_plane.cc
        address_info.cc
    )
endif()