# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(HCCL_CODE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(HCCL_LLT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(hccl_llt_stub
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_profiling_plugin.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_gdr.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_sal.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_fe.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_error_manager.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_mc2.cc
    # ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_GenRankTable.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_gert.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_platform.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_rts.cc
    ${HCCL_LLT_ROOT}/stub/llt_stub_sec.c
    ${HCCL_LLT_ROOT}/stub/llt_mmpa_stub.c
    ${HCCL_LLT_ROOT}/stub/llt_stub_mm.c
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_rts.cc
    ${HCCL_LLT_ROOT}/stub/legacy/interface_channel.cc
    ${HCCL_LLT_ROOT}/stub/llt_next_orion_stub.cc
    ${HCCL_LLT_ROOT}/stub/llt_next_hccp_stub.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_rank_graph.cc

    ${HCCL_LLT_ROOT}/stub/llt_next_orion_ccu_stub.cc

    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_orion_adapter.cc
    ${HCCL_CODE_ROOT}/src/legacy/framework/communicator/hostdpu/task_service.cc
    ${HCCL_CODE_ROOT}/src/legacy/framework/communicator/hostdpu/dpu_kernel_entrance.cc
)

file(GLOB_RECURSE ALGORITHM_SOURCES "${HCCL_CODE_ROOT}/src/algorithm/*.cc")
file(GLOB_RECURSE COMMON_SOURCES "${HCCL_CODE_ROOT}/src/common/*.cc")
file(GLOB_RECURSE FRAMEWORK_HOST_SOURCES "${HCCL_CODE_ROOT}/src/framework/*.cc")
file(GLOB_RECURSE FRAMEWORK_DEVICE_SOURCES "${HCCL_CODE_ROOT}/src/framework/*.cc")
file(GLOB_RECURSE HCCD_SOURCES "${HCCL_CODE_ROOT}/src/hccd/*.cc")
file(GLOB_RECURSE PLATFORM_SOURCES "${HCCL_CODE_ROOT}/src/platform/*.cc")

set(EXCLUDE_FREAMEWORK_DEVICE_REGEX
    "${HCCL_CODE_ROOT}/src/framework/communicator/impl/hccl_communicator_attrs_device.cc"
    "${HCCL_CODE_ROOT}/src/framework/communicator/impl/hccl_communicator_device.cc"
    "${HCCL_CODE_ROOT}/src/framework/common/src/config/env_config_aicpu.cc"
    "${HCCL_CODE_ROOT}/src/framework/communicator/hccl_comm_device.cc"
    "${HCCL_CODE_ROOT}/src/framework/next/coll_comms/api_c_adpt/dev/dev_coll_comm_c_adpt.cc"
    "${HCCL_CODE_ROOT}/src/framework/device/framework/zero_copy_address_mgr.cc"
    "${HCCL_CODE_ROOT}/src/framework/common/src/launch_aicpu.cc"
    "${HCCL_CODE_ROOT}/src/framework/op_base/src/op_base_device.cc"
    "${HCCL_CODE_ROOT}/src/framework/communicator/impl/independent_op/diag/aicpu/hcomm_diag_adapt_aicpu.cc"
    )
list(REMOVE_ITEM FRAMEWORK_HOST_SOURCES ${EXCLUDE_FREAMEWORK_DEVICE_REGEX})

set(EXCLUDE_ALGORITHM_DEVICE_REGEX 
    "${HCCL_CODE_ROOT}/src/algorithm/impl/hccl_alg_device.cc"
    "${HCCL_CODE_ROOT}/src/algorithm/impl/hccl_aiv_device.cc"
    "${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_executor_aicpu.cc"
    )
list(REMOVE_ITEM ALGORITHM_SOURCES ${EXCLUDE_ALGORITHM_DEVICE_REGEX})

set(EXCLUDE_COMMON_DEVICE_REGEX 
    "${HCCL_CODE_ROOT}/src/common/debug/profiling/plugin_runner_device.cc"
    "${HCCL_CODE_ROOT}/src/common/stream/stream_utils_aicpu.cc"
    "${HCCL_CODE_ROOT}/src/common/launch_device/launch_device.cc"
    )
list(REMOVE_ITEM COMMON_SOURCES ${EXCLUDE_COMMON_DEVICE_REGEX})

set(EXCLUDE_PLATFORM_DEVICE_REGEX
    "${HCCL_CODE_ROOT}/src/platform/resource/mem/mem_mapping_manager_device.cc"
    )
list(REMOVE_ITEM PLATFORM_SOURCES ${EXCLUDE_PLATFORM_DEVICE_REGEX})

list(FILTER PLATFORM_SOURCES EXCLUDE REGEX ".*src/platform/legacy/.*")

list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX ".*src/framework/next/coll_comms/rank_graphs/.*")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX ".*src/framework/next/comms/comm_engine_res/engine_ctxs/.*")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX ".*src/framework/next/comms/comm_engine_res/notifys/.*")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX ".*src/framework/next/comms/comm_engine_res/comm_engine_res.cc")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX ".*src/framework/next/comms/comm_engine_res/comm_engine_res_mgr.cc")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX ".*src/framework/next/comm_primitives/.*")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX "src/framework/communicator/impl/independent_op/data_api/sync_data_aicpu_ts.cc")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX "src/framework/common/src/topo/topo_device/topoinfo_ranktableParser_device.cc")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX "src/framework/communicator/impl/independent_op/data_api/hccl_api_data_aicpu_ts.cc")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX "src/framework/common/src/topo/topo_device/.*")

set(orion_list
    ${HCCL_CODE_ROOT}/src/framework/next/comms/comm_engine_res/threads/aicpu_ts_thread.cc
    ${HCCL_CODE_ROOT}/src/framework/next/comms/comm_engine_res/threads/thread.cc
    ${HCCL_CODE_ROOT}/src/framework/next/comms/comm_engine_res/threads/cpu_ts_thread.cc

    ${HCCL_CODE_ROOT}/src/legacy/interface/aicpu_ts_thread_interface.cc
    ${HCCL_CODE_ROOT}/src/legacy/common/exception/exception_defination.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/transport/aicpu/p2p_transport_lite_impl.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/buffer/buffer.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/buffer/aicpu/rma_buf_slice_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/notify/aicpu/notify_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/notify/rts_notify.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/notify/ipc_remote_notify.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/buffer/aicpu/rma_buffer_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/connection/aicpu/rma_conn_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/buffer/aicpu/rmt_rma_buf_slice_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/connection/aicpu/ub_conn_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/connection/aicpu/ub_conn_lite_mgr.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/connection/host_ub_connection.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/stream/stream.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/task/task.cc
    ${HCCL_CODE_ROOT}/src/legacy/interface/rank_graph_interface.cc
    ${HCCL_CODE_ROOT}/src/legacy/framework/topo/new_topo_builder/rank_graph/rank_graph.cc
    ${HCCL_CODE_ROOT}/src/legacy/framework/topo/new_topo_builder/rank_graph/net_instance.cc
    ${HCCL_CODE_ROOT}/src/legacy/framework/topo/virtual_topo.cc
    ${HCCL_CODE_ROOT}/src/legacy/framework/env_config/base_config.cc
    ${HCCL_CODE_ROOT}/src/legacy/framework/env_config/env_config.cc
    ${HCCL_CODE_ROOT}/src/legacy/framework/env_config/env_func.cc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/common/tp_qos.cc
)

set(hccl_list
    ${hccl_llt_stub}
    ${HCCD_SOURCES}
    ${COMMON_SOURCES}
    ${FRAMEWORK_HOST_SOURCES}
    ${ALGORITHM_SOURCES}
    ${PLATFORM_SOURCES}
    ${orion_list}
)

add_library(hccl_llt SHARED  ${hccl_list})

target_compile_definitions(hccl_llt PRIVATE
    OPEN_BUILD_PROJECT
)

#头文件搜索路径
target_include_directories(hccl_llt  PRIVATE
    ${HCCL_LLT_ROOT}/stub
    ${HCCL_LLT_ROOT}/stub/legacy/

    ${HCCL_CODE_ROOT}/src/framework/common/src
    ${HCCL_CODE_ROOT}/include/
    ${HCCL_CODE_ROOT}/include/hccl/
    ${HCCL_CODE_ROOT}/pkg_inc
    ${HCCL_CODE_ROOT}/pkg_inc/hccl
    ${HCCL_CODE_ROOT}/pkg_inc/hcomm/ccu
    ${HCCL_CODE_ROOT}/externel_depends/tsch
    ${HCCL_CODE_ROOT}/src/common/
    ${HCCL_CODE_ROOT}/src/pub_inc/
    ${HCCL_CODE_ROOT}/src/pub_inc/inner
    ${HCCL_CODE_ROOT}/src/pub_inc/aicpu
    ${HCCL_CODE_ROOT}/src/pub_inc/new
    ${HCCL_CODE_ROOT}/src/platform/hccp/inc
    ${HCCL_CODE_ROOT}/src/platform/hccp/inc/network
    ${HCCL_CODE_ROOT}/src/platform/hccp/inc/private/network
    ${RDMA_CORE_INCLUDE_DIR}
    ${HCCL_CODE_ROOT}/src/pkg_inc
    ${HCCL_CODE_ROOT}/external_depends

    # hccl/hccl目录下头文件
    ${HCCL_CODE_ROOT}/src/inc
    ${HCCL_CODE_ROOT}/src/algorithm/pub_inc
    ${HCCL_CODE_ROOT}/src/algorithm/impl
    ${HCCL_CODE_ROOT}/src/framework/
    ${HCCL_CODE_ROOT}/src/framework/inc/
    ${HCCL_CODE_ROOT}/src/framework/inc/host/
    ${HCCL_CODE_ROOT}/src/framework/cluster_maintenance/health/heartbeat/
    ${HCCL_CODE_ROOT}/src/framework/cluster_maintenance/detect/detect_connect_anomalies/
    ${HCCL_CODE_ROOT}/src/framework/cluster_maintenance/recovery/operator_retry
    ${HCCL_CODE_ROOT}/src/framework/cluster_maintenance/snapshot
    ${HCCL_CODE_ROOT}/src/framework/cluster_maintenance/aclgraph
    ${HCCL_CODE_ROOT}/src/framework/common/src/host
    ${HCCL_CODE_ROOT}/src/framework/common/src/config
    ${HCCL_CODE_ROOT}/src/framework/common/src/task
    ${HCCL_CODE_ROOT}/src/framework/common/src/topo
    ${HCCL_CODE_ROOT}/src/framework/common/src/exception
    ${HCCL_CODE_ROOT}/src/framework/common/src/thread
    ${HCCL_CODE_ROOT}/src/framework/common/src/onesided_memory_management
    ${HCCL_CODE_ROOT}/src/framework/communicator/impl
    ${HCCL_CODE_ROOT}/src/framework/communicator/impl/one_sided_service
    ${HCCL_CODE_ROOT}/src/framework/communicator/impl/resource_manager
    ${HCCL_CODE_ROOT}/src/framework/communicator/impl/independent_op
    ${HCCL_CODE_ROOT}/src/framework/communicator/impl/independent_op/channel/
    ${HCCL_CODE_ROOT}/src/framework/communicator/impl/independent_op/channel/device
    ${HCCL_CODE_ROOT}/src/framework/communicator/impl/independent_op/rank_graph
    ${HCCL_CODE_ROOT}/src/framework/communicator/impl/independent_op/resource/engine
    ${HCCL_CODE_ROOT}/src/framework/hcom/
    ${HCCL_CODE_ROOT}/src/framework/hcom/gradient_segment
    ${HCCL_CODE_ROOT}/src/framework/op_base/src/
    ${HCCL_CODE_ROOT}/src/framework/group
    ${HCCL_CODE_ROOT}/src/framework/nslbdp/
    ${HCCL_CODE_ROOT}/src/platform/
    ${HCCL_CODE_ROOT}/src/platform/common/
    ${HCCL_CODE_ROOT}/src/platform/common/unique
    ${HCCL_CODE_ROOT}/src/platform/common/unfold_cache
    ${HCCL_CODE_ROOT}/src/platform/common/buffer_manager
    ${HCCL_CODE_ROOT}/src/platform/inc
    ${HCCL_CODE_ROOT}/src/platform/inc/adapter
    ${HCCL_CODE_ROOT}/src/platform/inc/adapter/host
    ${HCCL_CODE_ROOT}/src/platform/inc/common/
    ${HCCL_CODE_ROOT}/src/platform/resource/transport/
    ${HCCL_CODE_ROOT}/src/platform/resource/transport/host/
    ${HCCL_CODE_ROOT}/src/platform/resource/transport/device/
    ${HCCL_CODE_ROOT}/src/platform/resource/transport/heterog/
    ${HCCL_CODE_ROOT}/src/platform/resource/transport/onesided/
    ${HCCL_CODE_ROOT}/src/platform/resource/transport/onesided/device
    ${HCCL_CODE_ROOT}/src/platform/resource/notify/
    ${HCCL_CODE_ROOT}/src/platform/resource/rma_buffer
    ${HCCL_CODE_ROOT}/src/platform/resource/socket
    ${HCCL_CODE_ROOT}/src/platform/resource/mem
    ${HCCL_CODE_ROOT}/src/platform/task/
    ${HCCL_CODE_ROOT}/src/platform/aiv_communication/
    ${HCCL_CODE_ROOT}/src/hccd

    ${HCCL_CODE_ROOT}/src/platform/resource/dispatcher_ctx

    # hccl/common 目录下头文件
    ${HCCL_CODE_ROOT}/src/common/health/
    ${HCCL_CODE_ROOT}/src/common/debug/profiling
    ${HCCL_CODE_ROOT}/src/common/debug/profiling/inc
    ${HCCL_CODE_ROOT}/src/common/debug/config
    ${HCCL_CODE_ROOT}/src/common/debug/profiling/inc/host
    ${HCCL_CODE_ROOT}/src/common/stream/
    ${HCCL_CODE_ROOT}/src/common/launch_device/

    # hccl/algorithm 目录下头文件
    ${HCCL_CODE_ROOT}/src/algorithm/base
    ${HCCL_CODE_ROOT}/src/algorithm/base/inc
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_aiv_template
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/inc_all_reduce_deter
    ${HCCL_CODE_ROOT}/src/algorithm/base/communicator
    ${HCCL_CODE_ROOT}/src/algorithm/base/communicator/legacy
    ${HCCL_CODE_ROOT}/src/algorithm/base/mc2_handler
    ${HCCL_CODE_ROOT}/src/algorithm/impl
    ${HCCL_CODE_ROOT}/src/algorithm/impl/inc
    ${HCCL_CODE_ROOT}/src/algorithm/impl/legacy
    ${HCCL_CODE_ROOT}/src/algorithm/impl/legacy/operator
    ${HCCL_CODE_ROOT}/src/algorithm/impl/resource_manager
    ${HCCL_CODE_ROOT}/src/algorithm/impl/task
    ${HCCL_CODE_ROOT}/src/algorithm/impl/operator
    ${HCCL_CODE_ROOT}/src/algorithm/impl/operator/registry
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/registry
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_all_gather
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_all_gather/310P
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_all_gather_v
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_all_gather_v/310P
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_all_reduce
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_all_reduce/310P
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_send_receive
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_reduce_scatter
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_reduce_scatter/310P
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_reduce_scatter_v
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_reduce_scatter_v/310P
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_all_to_all
    ${HCCL_CODE_ROOT}/src/algorithm/impl/coll_executor/coll_scatter

    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/temp_all_gather
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/temp_all_reduce
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/temp_alltoall
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/temp_alltoallv
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/temp_broadcast
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/temp_reduce
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/temp_reduce_scatter
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/temp_scatter
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template/component
    ${HCCL_CODE_ROOT}/src/algorithm/base/alg_template
    ${HCCL_CODE_ROOT}/src/algorithm/base/pub_inc

    # hccl/operator目录下头文件
    ${HCCL_CODE_ROOT}/operator

    # /hccl/framework/device/目录下头文件
    ${HCCL_CODE_ROOT}/src/framework/device/
    ${HCCL_CODE_ROOT}/src/framework/device/common
    ${HCCL_CODE_ROOT}/src/framework/device/inc
    ${HCCL_CODE_ROOT}/src/framework/device/aicpu_kfc
    ${HCCL_CODE_ROOT}/src/framework/device/aicpu_kfc/inc
    ${HCCL_CODE_ROOT}/src/framework/device/aicpu_kfc/common
    ${HCCL_CODE_ROOT}/src/framework/next/comms/comm_engine_res/threads

    # proto.h头文件
    ${CMAKE_BINARY_DIR}/proto/hccl_llt

    # tbe头文件
    ${HCCL_CODE_ROOT}/src/platform/legacy/hccl_tbe_task/
    ${HCCL_CODE_ROOT}/src/platform/legacy/common/
    ${HCCL_CODE_ROOT}/src/platform/legacy/graph_ctx_mgr/
    ${HCCL_CODE_ROOT}/src/platform/legacy/inc/

    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/
    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/aicpu/
    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/aicpu/common/
    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/runtime/
    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/profiling/
    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/base/
    ${HCCL_CODE_ROOT}/test/ut/depends/include/base/context_builder
    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/dump/
    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/trace/
    ${HCCL_CODE_ROOT}/test/ut/depends/pkg_inc/asc/include/adv_api/hccl/internal

    ${HCCL_CODE_ROOT}/test/ut/depends/include/
    ${HCCL_CODE_ROOT}/test/ut/depends/runtime/include/
    ${HCCL_CODE_ROOT}/test/ut/depends/include/mmpa/
    ${HCCL_CODE_ROOT}/test/ut/depends/include/driver/
    ${HCCL_CODE_ROOT}/test/ut/depends/include/acl/
    ${HCCL_CODE_ROOT}/test/ut/depends/include/ascendc/highlevel_api/
    ${HCCL_CODE_ROOT}/test/ut/depends/include/hcomm-legacy/
    ${HCCL_CODE_ROOT}/test/ut/depends/include/urma/

    ${THIRD_PARTY_NLOHMANN_PATH}
    ${CANN_3RD_LIB_PATH}/json/include/
    ${CANN_3RD_LIB_PATH}/gtest/include/

    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/transport/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/stream/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/socket
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/common/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/channels
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/channels/aicpu
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/channels/host
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/channels/host/dpu_notify
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoints
    ${HCCL_CODE_ROOT}/src/framework/next/comms/api_c_adpt
    ${HCCL_CODE_ROOT}/src/framework/next/comms/common
    ${HCCL_CODE_ROOT}/src/framework/next/comms/common/device
    ${HCCL_CODE_ROOT}/src/framework/next/common
    ${HCCL_CODE_ROOT}/src/legacy/common
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/stream/aicpu/
    ${HCCL_CODE_ROOT}/src/legacy/common/types
    ${HCCL_CODE_ROOT}/src/legacy/common/utils
    ${HCCL_CODE_ROOT}/src/legacy/common/exception/
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/external_system
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/common
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/buffer
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/transport/
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/pub_inc
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/
    ${HCCL_CODE_ROOT}/src/legacy/framework/env_config
    ${HCCL_CODE_ROOT}/src/legacy/service/collective/primitive
    ${HCCL_CODE_ROOT}/src/legacy/framework/communicator/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/framework/communicator/aicpu/inc
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/sockets
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/notify/
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/notify/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/mem
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/mem
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/connection/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/socket/
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/resource/buffer/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/unified_platform/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/legacy/service/collective
    ${HCCL_CODE_ROOT}/src/legacy/framework/communicator
    ${HCCL_CODE_ROOT}/src/legacy/interface
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/

    #next
    ${HCCL_CODE_ROOT}/src/framework/next/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/comm_engine_res/threads
    ${HCCL_CODE_ROOT}/src/framework/next/comms/comm_engine_res/threads/slaves/
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/communicator
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/communicator/aicpu
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/rank
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/rank/comm_mems/
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/rank_pairs
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/common/loggers
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/channels
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/channels/aicpu/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/channels/aicpu/device/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/sockets/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoints/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoints/dfx
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/api_c_adpt/


    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoints/server_socket/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoints/reged_mems/

    ${HCCL_CODE_ROOT}/src/framework/next/comms/adpt/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/pub_inc/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_device/ccu_comp/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_device/ccu_comp/ccu_channel/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1/
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_transport/
 
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_kernel

    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_microcode
 
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/interface
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/context
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/reps
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/reps/arithmetic
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/reps/common
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/reps/control
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/reps/data
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/reps/loop
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/reps/sync
    ${HCCL_CODE_ROOT}/src/framework/next/comms/ccu/ccu_representation/reps/translator
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/dfx
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/dfx/cluster_monitor
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/dfx/profiling
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/dfx/profiling/aicpu
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/dfx/profiling/host
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/dfx/taskException
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/dfx/taskException/aicpu
    ${HCCL_CODE_ROOT}/src/framework/next/coll_comms/dfx/taskException/host
    ${HCCL_CODE_ROOT}/src/framework/device/framework
    ${HCCL_CODE_ROOT}/src/framework/next/comms/endpoint_pairs/channels/slaves
    ${HCCL_CODE_ROOT}/src/framework/dfx/task_exception
    ${HCCL_CODE_ROOT}/src/legacy/framework/dfx/task_exception
    ${HCCL_CODE_ROOT}/src/legacy/framework/dfx/common
    ${HCCL_CODE_ROOT}/src/legacy/framework/dfx/profiling
    ${HCCL_CODE_ROOT}/src/legacy/framework/dfx/
    ${HCCL_CODE_ROOT}/src/legacy/framework/resource_manager/stream
    ${HCCL_CODE_ROOT}/src/legacy/interface/

)

FILE(GLOB_RECURSE sources_list LIST_DIRECTORIES TRUE ${HCCL_CODE_ROOT}/src/legacy)
foreach (dir ${sources_list})
    IF ((IS_DIRECTORY ${dir}) AND (${dir} MATCHES "src/legacy") AND (NOT ${dir} MATCHES "src/legacy/mdpi"))
        target_include_directories(hccl_llt  PRIVATE  ${dir})
    endif()
endforeach()

target_compile_definitions(hccl_llt    PRIVATE
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    -DCCL_LLT
)

target_compile_options(hccl_llt   PRIVATE
    -O0 -U_FORTIFY_SOURCE -g
)

target_link_libraries(hccl_llt  PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -lrt
    -ldl
)
