if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        rank_info_detect_service.cc
        rank_info_detect_client.cc
        rank_info_detect.cc
        bootstrap_ip.cc
        preempt_port_manager.cc
        rank_info_dispatcher.cc
        socket_agent.cc
    )
endif()