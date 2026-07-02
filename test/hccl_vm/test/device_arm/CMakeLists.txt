# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.14)

add_subdirectory(proxy)

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
    ${CMAKE_SOURCE_DIR}/third_party
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

# 1. test_hccl_task_collection (10 TEST_F)
add_executable(test_hccl_task_collection hccl_task_collection_test.cc)

target_sources(test_hccl_task_collection PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_task_collection.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_hccl_task_collection PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_hccl_task_collection PRIVATE ${COMMON_LIBS})

set_target_properties(test_hccl_task_collection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 2. test_hccl_device_pub (12 TEST_F)
add_executable(test_hccl_device_pub hccl_device_pub_test.cc)

target_sources(test_hccl_device_pub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_sqlite_db.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_hccl_device_pub PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_hccl_device_pub PRIVATE ${COMMON_LIBS})

set_target_properties(test_hccl_device_pub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

target_compile_options(test_hccl_device_pub PRIVATE
    -DCANN_VERSION_NUM=90000000
)

# 3. test_hccl_kernel_executor (21 TEST_F)
add_executable(test_hccl_kernel_executor hccl_kernel_executor_test.cc)

target_sources(test_hccl_kernel_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_kernel_executor.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_sqlite_db.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_hccl_kernel_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/test/stub_headers
    ${COMMON_INCLUDE_DIRS}
)

target_link_libraries(test_hccl_kernel_executor PRIVATE ${COMMON_LIBS})

set_target_properties(test_hccl_kernel_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# CANN
set(HCCL_CANN_VERSION_NUM 0)
set(_hccl_cann_version_header "${ASCEND_CANN_PACKAGE_PATH}/include/version/cann_version.h")
message(STATUS "Checking HCCL scoped CANN version header: ${_hccl_cann_version_header}")
if(EXISTS "${_hccl_cann_version_header}")
    file(STRINGS "${_hccl_cann_version_header}" _hccl_cann_ver_line
        REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_hccl_cann_ver_line)
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
            CANN_EXPR "${_hccl_cann_ver_line}")
        math(EXPR HCCL_CANN_VERSION_NUM "${CANN_EXPR}" OUTPUT_FORMAT DECIMAL)
    endif()
endif()

message(STATUS "HCCL_CANN_VERSION_NUM: ${HCCL_CANN_VERSION_NUM}")
if(HCCL_CANN_VERSION_NUM EQUAL 0)
    message(WARNING "HCCL failed to resolve CANN_VERSION_NUM (got 0). "
        "Compatibility branches (e.g. HCCL_CANN_COMPAT_850) will fall back to the 9.0.0+ path.")
endif()

target_compile_options(test_hccl_kernel_executor PRIVATE
    -DCANN_VERSION_NUM=${HCCL_CANN_VERSION_NUM}
)
