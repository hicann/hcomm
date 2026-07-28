if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        net_instance.cc
        rank_graph.cc
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        net_instance.cc
        rank_graph.cc
    )
endif()