# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.14)
project(hcc_vm_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

add_subdirectory(cmd)
add_subdirectory(device_arm)
add_subdirectory(plugin)
add_subdirectory(proxy)
add_subdirectory(runnerdb)
add_subdirectory(store)
add_subdirectory(src_root)
add_subdirectory(hccl_plugin_manager)
add_subdirectory(utils)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/stub_headers
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/adv_api
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/include/store
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_cmd
    ${HCOMM_VM_ROOT_PATH}/src/proxy
)

enable_testing()
add_test(NAME hccl_host_test COMMAND hccl_host_test)
