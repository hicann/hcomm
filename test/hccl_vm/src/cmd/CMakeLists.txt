# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

file(GLOB HCCL_CMD_SOURCES CONFIGURE_DEPENDS
        "*.cc"
        "subcmd/*.cc"
)

add_library(cmd OBJECT ${HCCL_CMD_SOURCES})

target_include_directories(cmd PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hcomm/ccu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/subcmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/../store
        ${CMAKE_CURRENT_SOURCE_DIR}/../store
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/runnerdb
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/topo
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/store
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/nlohmann_json

        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
)

target_compile_options(cmd PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(cmd PRIVATE
        dl
        rt
        log
        runnerdb
        store
        topo
        CLI11::CLI11
        yaml-cpp::yaml-cpp
        JSON::JSON
)

if(TARGET third_party_yaml_cpp)
    add_dependencies(cmd third_party_yaml_cpp)
endif()

# 日志染色: 模块 tag
target_compile_definitions(cmd PRIVATE HCCL_VM_MODULE="CMD")
