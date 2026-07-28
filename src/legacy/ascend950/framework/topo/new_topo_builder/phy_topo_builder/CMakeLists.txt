if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
            phy_topo_builder.cc
    )
endif()