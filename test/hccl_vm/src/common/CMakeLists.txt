# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

file(GLOB_RECURSE HCCL_SOURCES "*.cc")

add_library(common STATIC ${HCCL_SOURCES})

target_compile_options(common PRIVATE
    -fPIC
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_include_directories(common PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/runnerdb
    ${PROJECT_SOURCE_DIR}/include/store
    ${PROJECT_SOURCE_DIR}/include

    ${HCCL_VM_SRC}
    ${HCCL_VM_SRC}/common
    ${HCCL_VM_SRC}/plugin
    ${HCCL_VM_SRC}/cmd
    ${HCCL_VM_SRC}/store
    ${HCCL_VM_SRC}/store
    ${HCCL_VM_SRC}/runnerdb
    ${THRID_PARTY_DIR}/nlohmann_json

    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/toolchain

    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts
)

target_link_libraries(common PRIVATE
    log
    runnerdb
)
