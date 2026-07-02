# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.14)

set(AIV_EXECUTOR_SOURCES
    aiv_resource_manager.cc
    aiv_graph_executor.cc
    aiv_graph_executor_mgr.cc
    aiv_task_snapshot_loader.cc
    ${PROJECT_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub/ascendc_base_stub.cc
    ${PROJECT_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/ai_core_stub.cc
    ${PROJECT_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task.cc
)

add_library(aiv_executor STATIC ${AIV_EXECUTOR_SOURCES})

target_compile_options(aiv_executor PRIVATE
    -g
    -O0
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_include_directories(aiv_executor
    PUBLIC
        ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hcomm/ccu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HCCL_VM_SRC}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/runnerdb
        ${PROJECT_SOURCE_DIR}/include/store
        ${PROJECT_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub
        ${PROJECT_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub

        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    PRIVATE
        ${HCCL_VM_SRC}/store
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling

        ${THRID_PARTY_DIR}/nlohmann_json
        ${THRID_PARTY_DIR}/dbg_macro   # for debug
)

target_link_libraries(aiv_executor
    PRIVATE
        store
        log
        rt
)

# 日志染色: 模块 tag
target_compile_definitions(aiv_executor PRIVATE HCCL_VM_MODULE="RUNNER")
