# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 hccl_v2 动态链接库，在 host 侧使用
add_library(hccl_v2 SHARED)

# 宏定义
target_compile_definitions(hccl_v2 PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
)

# 编译选项
target_compile_options(hccl_v2 PRIVATE
    -Werror
    -Wfloat-equal
    -Wall
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O3
    -std=c++17
    -fstack-protector-all
    $<$<CONFIG:Debug>:-g>
)

# 链接选项
target_link_options(hccl_v2 PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    -Wl,--build-id=none
    $<$<CONFIG:Release>:-s>
)

# 链接库
target_link_libraries(hccl_v2 PRIVATE
    $<BUILD_INTERFACE:acl_rt_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:atrace_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<BUILD_INTERFACE:json>
    -Wl,--no-as-needed
    c_sec
    unified_dlog
    mmpa
    runtime
    acl_rt
    error_manager
    ccl_dpu
    tsdclient
    ra
    -Wl,--as-needed
    hccl_headers
    topoaddrinfo
)

target_include_directories(hccl_v2 PRIVATE
    # src/legacy 头文件
    ${LEGACY_ASCEND950_INCLUDE_LIST}
    # 内部头文件
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl/
    ${HCOMM_DIR}/pkg_inc
    # pub_inc 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/common
    # src/algorithm 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator
    # src/platform 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter
    # hccp 头文件 (moved to base_comm/resources)
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_DIR}/src/base_comm/resources/hccp/orion/hcomm_dev/inc/network
    # base_comm 公共头文件
    ${HCOMM_DIR}/src/base_comm/common
    # 外部依赖
    ${HCOMM_DIR}/externel_depends/tsch
)

# 将hccl编译出的动态库加入CANN的安装包
install(TARGETS hccl_v2
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
