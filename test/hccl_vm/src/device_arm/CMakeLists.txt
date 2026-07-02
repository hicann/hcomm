# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.10)
project(device)

# 设置交叉编译器
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
# 设置交叉编译工具链路径
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

if(NOT DEFINED HCCL_VM_SRC)
    set(HCCL_VM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
if(NOT DEFINED THRID_PARTY_DIR)
    set(THRID_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)
endif()
if(NOT DEFINED HCCL_VM_INCLUDE)
    set(HCCL_VM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
endif()
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/device-install" CACHE PATH "..." FORCE)
endif()
if(NOT DEFINED HCCL_CODE_ROOT)
    set(HCCL_CODE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(proxy)

add_library(sqlite3 STATIC "")
target_sources(sqlite3 PRIVATE
    ${THRID_PARTY_DIR}/sqlite/sqlite3.c
)
target_include_directories(sqlite3 PRIVATE
    ${THRID_PARTY_DIR}/sqlite
)

add_executable(device
    main.cc
    hccl_device_pub.cc
    hccl_task_collection.cc
    hccl_kernel_executor.cc
    ${HCCL_VM_SRC}/store/store_sim_memory_manager.cc
    ${HCCL_VM_SRC}/store/store_sim_shm_ops.cc
    ${HCCL_VM_SRC}/runnerdb/tracedb/db_sim_sqlite_db.cc
    ${HCCL_VM_SRC}/utils/sim_log.cc
    ${HCCL_VM_SRC}/utils/sim_yaml_config.cc
    ${HCCL_VM_SRC}/utils/sim_common_api.cc
    ${HCCL_VM_SRC}/proxy/log_stub.cc
    ${HCCL_VM_SRC}/runnerdb/tracedb/db_sim_op_db_ops.cc
    ${HCCL_VM_SRC}/runnerdb/modeldb/db_hccl_op_db_ops.cc
    ${HCCL_VM_SRC}/runnerdb/modeldb/db_hccl_db_factory.cc
    ${HCCL_VM_SRC}/runnerdb/modeldb/db_hccl_db_sqlite.cc
)

target_link_libraries(device PRIVATE
    slog
    aicpu_sharder
    sqlite3
    -ldl
    -lrt
)

target_include_directories(device PRIVATE
    ${HCCL_CODE_ROOT}/src/ops/op_common/inc
    ${HCCL_CODE_ROOT}/src/common
    ${HCCL_CODE_ROOT}/src/common/hcomm_dlsym
    ${HCCL_CODE_ROOT}/src/common/hcomm_dlsym/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/hcomm/ccu

    ${HCCL_VM_INCLUDE}
    ${HCCL_VM_INCLUDE}/store
    ${HCCL_VM_INCLUDE}/runnerdb

    ${HCCL_VM_SRC}/utils
    ${HCCL_VM_SRC}/device_arm
    ${HCCL_VM_SRC}/runnerdb/modeldb
    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/spdlog
)

# 自适应CANN版本号变更，从CANN包version头解析CANN_VERSION_NUM
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

target_compile_options(device PRIVATE
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
    -DCANN_VERSION_NUM=${HCCL_CANN_VERSION_NUM}
    -DNO_YAML_CONFIG
)

target_link_options(device PUBLIC "-Wl,--export-dynamic")

add_dependencies(device sqlite3)

install(TARGETS device
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib/aarch64
    ARCHIVE DESTINATION lib/aarch64
)



# 日志染色: 模块 tag
target_compile_definitions(device PRIVATE HCCL_VM_MODULE="DEVICE")
