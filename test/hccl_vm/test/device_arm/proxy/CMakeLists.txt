# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.14)

set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/device_arm
    ${CMAKE_SOURCE_DIR}/src/device_arm/proxy
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/third_party/CLI11
    ${CMAKE_BINARY_DIR}/_deps/yaml-cpp-src/include
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    $ENV{HCCL_CODE_HOME}/src/ops/op_common/inc
    $ENV{HCCL_CODE_HOME}/src/common
    $ENV{HCCL_CODE_HOME}/src/common/hcomm_dlsym
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/include/driver
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/pkg_inc/driver
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/pkg_inc/runtime
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/pkg_inc/toolchain
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/pkg_inc/profiling
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/pkg_inc
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/pkg_inc/hccl
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/pkg_inc/base
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/include
    $ENV{ASCEND_TOOLKIT_HOME}/x86_64-linux/include/hccl
)

set(COMMON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils/sim_log.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/log_stub.cc
)

set(COMMON_LIBS
    gtest_main
    gtest
    log
    dl
    rt
    store
    runnerdb
    sqlite3
)

# test_device_sqe_parse_stub (89 TEST_F)
add_executable(test_device_sqe_parse_stub device_sqe_parse_stub_test.cc)

target_sources(test_device_sqe_parse_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/proxy/device_sqe_parse_stub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_sqlite_db.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_device_sqe_parse_stub PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_device_sqe_parse_stub PRIVATE ${COMMON_LIBS})

set_target_properties(test_device_sqe_parse_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

target_compile_options(test_device_sqe_parse_stub PRIVATE
    -DCANN_VERSION_NUM=90000000
)

# test_ascend_hal_stub (36 TEST_F)
add_executable(test_ascend_hal_stub ascend_hal_stub_test.cc)

target_sources(test_ascend_hal_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/proxy/ascend_hal_stub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/proxy/device_sqe_parse_stub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_sqlite_db.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_runner_common.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_ascend_hal_stub PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_ascend_hal_stub PRIVATE ${COMMON_LIBS})

set_target_properties(test_ascend_hal_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
