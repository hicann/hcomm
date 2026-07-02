# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_subdirectory(utils)
add_subdirectory(common)
add_subdirectory(proxy)
add_subdirectory(runnerdb)
add_subdirectory(store)
add_subdirectory(plugin)
add_subdirectory(cmd)
add_subdirectory(topo)

add_executable(hccl-vm 
    main.cc
    plugin/hccl_plugin_manager.cc
    plugin/hccl_plugin.cc
)

message(STATUS "ASCEND_CANN_PACKAGE_PATH12 = ${ASCEND_CANN_PACKAGE_PATH}")
target_include_directories(hccl-vm PUBLIC
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
    # hccl vm头文件
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/src/store
    ${HCOMM_VM_ROOT_PATH}/src/dep
    ${HCOMM_VM_ROOT_PATH}/src/host
    ${HCOMM_VM_ROOT_PATH}/src/host/hccl_cmd
    ${HCOMM_VM_ROOT_PATH}/src/host/hccl_plugin
    ${HCOMM_VM_ROOT_PATH}/src/runnerdb
    ${HCOMM_VM_ROOT_PATH}/src/cmd

    ${HCCL_VM_SRC}/store

    ## cann包头文件
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/toolchain

    # cann包头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts
    

    # hccl_vm头文件
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/include/store
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_cmd
    ${HCOMM_VM_ROOT_PATH}/src/proxy

    # hccl_proxy模块内部头文件
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/topomodel
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)

target_link_libraries(hccl-vm
    PRIVATE
    -Wl,--whole-archive
    cmd
    common
    store
    topo
    loader
    runnerdb
    modeldb
    log
    -Wl,--no-whole-archive
    dl
    rt
    sqlite3
    yaml-cpp::yaml-cpp
    CLI11::CLI11
    JSON::JSON
)

target_compile_options(hccl-vm PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_include_directories(hccl-vm PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
)

set(JSON_REQ_VERSION "3.11.3")

target_link_libraries(hccl-vm PRIVATE CLI11::CLI11)
target_link_libraries(hccl-vm PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(hccl-vm PRIVATE JSON::JSON)

if(TARGET third_party_yaml_cpp)
    add_dependencies(hccl-vm third_party_yaml_cpp)
endif()

install(TARGETS hccl-vm
    DESTINATION "bin"
    # 文件的权限 (755)
    PERMISSIONS 
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)
