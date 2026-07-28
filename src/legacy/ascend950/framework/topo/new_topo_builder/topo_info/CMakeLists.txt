if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        edge_info.cc
        peer_info.cc
        topo_info.cc
    )
endif()