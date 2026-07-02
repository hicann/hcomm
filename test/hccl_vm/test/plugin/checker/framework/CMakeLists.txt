# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/singletask_check
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/third_party/CLI11
    ${CMAKE_BINARY_DIR}/_deps/yaml-cpp-src/include
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${CMAKE_SOURCE_DIR}/third_party/spdlog
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
)

set(CHECKER_LOG_SRC ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log/log_stub.cc)
set(CHECKER_LIBC_SEC ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64/libc_sec.so)

link_libraries(
    gtest
    log
    ${CHECKER_LIBC_SEC}
)

# Semantics Check Tests
add_executable(test_allgather_semantics_checker semantics_check/allgather_semantics_checker_test.cc main.cc)
target_sources(test_allgather_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/allgather_semantics_checker.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_allgather_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_allgather_v_semantics_checker semantics_check/allgather_v_semantics_checker_test.cc main.cc)
target_sources(test_allgather_v_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/allgather_v_semantics_checker.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_allgather_v_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_allreduce_semantics_checker semantics_check/allreduce_semantics_checker_test.cc main.cc)
target_sources(test_allreduce_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/allreduce_semantics_checker.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_json_utils.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_allreduce_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_broadcast_semantics_checker semantics_check/broadcast_semantics_checker_test.cc main.cc)
target_sources(test_broadcast_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/broadcast_semantics_checker.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_broadcast_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_reduce_semantics_checker semantics_check/reduce_semantics_checker_test.cc main.cc)
target_sources(test_reduce_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/reduce_semantics_checker.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_json_utils.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_reduce_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_scatter_semantics_checker semantics_check/scatter_semantics_checker_test.cc main.cc)
target_sources(test_scatter_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/scatter_semantics_checker.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_scatter_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_send_recv_semantics_checker semantics_check/send_recv_semantics_checker_test.cc main.cc)
target_sources(test_send_recv_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/send_recv_semantics_checker.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_send_recv_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_all2all_semantics_checker semantics_check/all2all_semantics_checker_test.cc main.cc)
target_sources(test_all2all_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/all2all_semantics_checker.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_all2all_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_reduce_scatter_semantics_checker semantics_check/reduce_scatter_semantics_checker_test.cc main.cc)
target_sources(test_reduce_scatter_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/reduce_scatter_semantics_checker.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_json_utils.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_reduce_scatter_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_reduce_scatter_v_semantics_checker semantics_check/reduce_scatter_v_semantics_checker_test.cc main.cc)
target_sources(test_reduce_scatter_v_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/reduce_scatter_v_semantics_checker.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_json_utils.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_reduce_scatter_v_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_batchsendrecv_semantics_checker semantics_check/batchsendrecv_semantics_checker_test.cc main.cc)
target_sources(test_batchsendrecv_semantics_checker PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/batchsendrecv_semantics_checker.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_batchsendrecv_semantics_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# Singletask Check Tests
add_executable(test_singletask_check singletask_check/singletask_check_test.cc main.cc)
target_sources(test_singletask_check PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/singletask_check/singletask_check.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/sim_task.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_ccu.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/check_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/storage_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/binary_data_operator.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator/ccu_all_rank_param_recorder.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_singletask_check BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/checker
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)
get_target_property(_singletask_includes test_singletask_check INCLUDE_DIRECTORIES)
list(FILTER _singletask_includes EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/include$")
set_target_properties(test_singletask_check PROPERTIES INCLUDE_DIRECTORIES "${_singletask_includes}")
target_link_libraries(test_singletask_check PRIVATE
    ${CHECKER_LIBC_SEC}
)
set_target_properties(test_singletask_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# Mem Conflict Check Tests
add_executable(test_check_rank_mem mem_conflict_check/check_rank_mem_test.cc main.cc)
target_sources(test_check_rank_mem PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mem_conflict_check/gen_ccu_task_node_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/check_rank_mem.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/mem_conflict_check_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/task_graph_revamp_base.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/task_graph_revamp_parallel.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/sim_task.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_ccu.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/check_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/binary_data_operator.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/storage_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator/ccu_all_rank_param_recorder.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_check_rank_mem BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/checker
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/singletask_check
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/dfx
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)
get_target_property(_check_rank_mem_includes test_check_rank_mem INCLUDE_DIRECTORIES)
list(FILTER _check_rank_mem_includes EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/include$")
set_target_properties(test_check_rank_mem PROPERTIES INCLUDE_DIRECTORIES "${_check_rank_mem_includes}")
target_link_libraries(test_check_rank_mem PRIVATE
    ${CHECKER_LIBC_SEC}
)
set_target_properties(test_check_rank_mem PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_mem_conflict_check_utils mem_conflict_check/mem_conflict_check_utils_test.cc main.cc)
target_sources(test_mem_conflict_check_utils PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/mem_conflict_check_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/sim_task.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_ccu.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_string_util.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_mem_conflict_check_utils PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/checker
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
)
target_link_libraries(test_mem_conflict_check_utils PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64/libc_sec.so
)
set_target_properties(test_mem_conflict_check_utils PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# test_task_graph_revamp_base
add_executable(test_task_graph_revamp_base mem_conflict_check/task_graph_revamp_base_test.cc main.cc)
target_sources(test_task_graph_revamp_base PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/task_graph_revamp_base.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/sim_task.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_ccu.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_string_util.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_task_graph_revamp_base PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)
target_link_libraries(test_task_graph_revamp_base PRIVATE
    ${CHECKER_LIBC_SEC}
)
set_target_properties(test_task_graph_revamp_base PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_task_graph_revamp mem_conflict_check/task_graph_revamp_test.cc main.cc)
target_sources(test_task_graph_revamp PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/task_graph_revamp.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/sim_task.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_task_graph_revamp PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/checker
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)
target_link_libraries(test_task_graph_revamp PRIVATE
    ${CHECKER_LIBC_SEC}
)
set_target_properties(test_task_graph_revamp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_task_graph_revamp_parallel mem_conflict_check/task_graph_revamp_parallel_test.cc main.cc)
target_sources(test_task_graph_revamp_parallel PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/task_graph_revamp_parallel.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/mem_conflict_check/task_graph_revamp_base.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/sim_task.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_ccu.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_string_util.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_task_graph_revamp_parallel PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)
target_link_libraries(test_task_graph_revamp_parallel PRIVATE
    ${CHECKER_LIBC_SEC}
)
set_target_properties(test_task_graph_revamp_parallel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_ccu_all_rank_param_recorder task_graph_generator/ccu_all_rank_param_recorder_test.cc main.cc)
target_sources(test_ccu_all_rank_param_recorder PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator/ccu_all_rank_param_recorder.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator/type_conversion.cc
    ${CHECKER_LOG_SRC}
)
set_target_properties(test_ccu_all_rank_param_recorder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_task_graph_generator task_graph_generator/task_graph_generator_test.cc task_graph_generator/ccu_task_transform_stub.cc main.cc)
target_sources(test_task_graph_generator PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator/task_graph_generator.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator/ccu_task_transform_instruct_common.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/sim_task.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_graph.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_json_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/validation_issue_recorder.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/file_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/setting_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/task_ccu.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_run_manifest.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/msgpack_writer.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/file_utils.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/storage_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/binary_data_operator.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator/ccu_all_rank_param_recorder.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_task_graph_generator BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/checker
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)
target_link_libraries(test_task_graph_generator PRIVATE
    ${CHECKER_LIBC_SEC}
)
set_target_properties(test_task_graph_generator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
