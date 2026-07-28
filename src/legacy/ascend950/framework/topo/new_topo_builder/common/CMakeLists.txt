if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        conn_interface.cc
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        conn_interface.cc
    )
endif()