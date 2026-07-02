# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.10)

project(checker_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 创建checker.bin
add_executable(checker
    main.cc
)

add_subdirectory(src)

# 如果需要暴露头文件目录
target_include_directories(checker PUBLIC
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/checker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/mem_conflict_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/semantics_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/singletask_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/task_graph_generator
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dfx

    ${CMAKE_CURRENT_SOURCE_DIR}/header/external
    ${CMAKE_CURRENT_SOURCE_DIR}/header/internal

    # hcomm仓头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/../hcomm/pkg_inc

    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
)

target_sources(checker PRIVATE
    ${HCCL_VM_INCLUDE}/sim_log.h
)

target_link_libraries(checker PRIVATE 
    log
    loader
    runnerdb
    modeldb
    store
    common
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64/libc_sec.so
    pthread
    yaml-cpp::yaml-cpp
    sqlite3
)

set_target_properties(checker PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(checker PRIVATE -g)

target_compile_options(checker PRIVATE
       -std=c++14
       -fno-common
       -Werror
       -fno-strict-aliasing
       -pipe
       -O0
       -g
       -D_GLIBCXX_USE_CXX11_ABI=0
)

install(TARGETS checker
    DESTINATION "plugin/checker"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
 )

install(FILES "manifest.json"
    DESTINATION "plugin/checker"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

# 日志染色: 模块 tag
target_compile_definitions(checker PRIVATE HCCL_VM_MODULE="CHECKER")
