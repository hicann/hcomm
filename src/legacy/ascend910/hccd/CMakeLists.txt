# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(PRODUCT_SIDE STREQUAL "host")
    return()
endif()

# 定义 hccd 动态链接库，在 device 侧使用
add_library(hccd SHARED)

# 宏定义
target_compile_definitions(hccd PRIVATE
    HCCD
)

# 编译选项
target_compile_options(hccd PRIVATE
    -Werror
    -Wno-deprecated-declarations
    -Wall
    -fno-common
    -fstack-protector-strong
    -fno-strict-aliasing
    -pipe
    -O3
    -std=c++17
    $<$<CONFIG:Debug>:-g>
)

# 链接选项
target_link_options(hccd PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

# 源文件列表
target_sources(hccd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/hccd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccd_impl_pml.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccd_comm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_comm_conn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_comm_conn_mgr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_raw.cc

    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager/workspace_resource.cc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager/workspace_resource_impl.cc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager/workspace_mem.cc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager/offload_stream_manager.cc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager/ccl_buffer_manager.cc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager/share_ccl_buffer_manager.cc

    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/config.cc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/json_utils.cc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/param_check.cc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo/topoinfo_ranktableParser.cc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo/topoinfo_ranktableConcise.cc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo/topoinfo_ranktableHeterog.cc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo/topoinfo_ranktableStandard.cc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo/topoinfo_roletableParser.cc

    ${HCOMM_DIR}/src/legacy/ascend910/common/health/rank_consistentcy_checker.cc
    ${HCOMM_DIR}/src/legacy/ascend910/common/health/calc_crc.cc
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/config/config_log.cc
)

# 头文件搜索路径
target_include_directories(hccd PRIVATE
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/hccl

    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/inner
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/aicpu

    # src/common 头文件
    ${HCOMM_DIR}/src/legacy/ascend910/common/health
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc/host
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/config

    # src/framework 头文件
    ${HCOMM_DIR}/src/legacy/ascend910/framework
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/config
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/exception
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/task
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/mgr
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/one_sided_service
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/zero_copy
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/independent_op
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graphs
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/independent_op/resource/engine
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/independent_op/channel
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/independent_op/channel/device
    ${HCOMM_DIR}/src/base_comm/resources/endpoints
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
    ${HCOMM_DIR}/src/base_comm/resources/reged_mems
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/sockets
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads

    # src/platform 头文件
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common
    ${HCOMM_DIR}/src/legacy/ascend910/platform/task
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_DIR}/src/legacy/ascend910/platform/ping_mesh
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/device
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/host
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/heterog
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/onesided
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/onesided/device
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/notify
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/socket
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/dispatcher_ctx
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/heterog

    # src/algorithm 头文件
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/legacy/operator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_aiv_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/component
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/mc2_handler
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_alltoall
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_alltoallv
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_reduce
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_scatter
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template

    ${HCOMM_DIR}/externel_depends/tsch/

    # src/legacy 头文件
    ${LEGACY_ASCEND950_INCLUDE_LIST}
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccd PRIVATE
        OPEN_BUILD_PROJECT
    )

    target_include_directories(hccd PRIVATE
        # 三方件头文件
        ${JSON_INCLUDE_DIR}
        ${CANN_3RD_LIB_PATH}/hcomm_utils/${PRODUCT_SIDE}/include/legacy
        ${HCOMM_DIR}/src/legacy/ascend910/common
        ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new/
        ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    )

    target_link_libraries(hccd
        $<BUILD_INTERFACE:ascend_hal_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:rdma_core_headers>
        $<BUILD_INTERFACE:hccl_legacy_headers>
        $<BUILD_INTERFACE:json>
        -Wl,--no-as-needed
        c_sec
        mmpa
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
    )
else()
    target_include_directories(hccd PRIVATE
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/aicpu
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/pkg_inc/aicpu_sched
        ${TOP_DIR}/runtime/pkg_inc/profiling
        ${TOP_DIR}/runtime/pkg_inc/runtime
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/asc/asc-devkit/include/adv_api/hccl/internal
        ${TOP_DIR}/asc/asc-devkit/include/adv_api/hccl/internal/hcomm
        ${HCOMM_DIR}/src/legacy/ascend910/common
        ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new/
        ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    )

    target_link_libraries(hccd
        $<BUILD_INTERFACE:intf_pub_cxx14>
        $<BUILD_INTERFACE:hccl_headers>
        -Wl,--no-as-needed
        c_sec
        slog
        mmpa
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
        ofed_headers
    )

    # 安装
    install(TARGETS hccd
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()
