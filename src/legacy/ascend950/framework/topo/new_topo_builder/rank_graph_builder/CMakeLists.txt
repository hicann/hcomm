if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        rank_graph_builder.cc
        detour_rules.cc
        detour_service.cc
        updater_for_64_plus_1.cc
    )
endif()