# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(TLV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tlv)

set(rs_list
    rs.c
    rs_epoll.c
    rs_ping.c
    rs_ping_inner.c
    rs_ping_roce.c
    rs_socket.c
    rs_rdma.c
    rs_nda.c
    rs_comm.c
    rs_drv_sensor.c
    rs_drv_rdma.c
    rs_drv_socket.c
    dl_ibverbs_function.c
    dl_nda_function.c
    hccp_dl.c
    ctx/dl_net_function.c
    ctx/dl_ccu_function.c
    ctx/dl_urma_function.c
    ctx/rs_ping_urma.c
    ctx/rs_ccu.c
    ctx/rs_ctx.c
    ctx/rs_esched.c
    ctx/rs_ub.c
    ctx/rs_ub_dfx.c
    ctx/rs_ub_tp.c
    ctx/product/ascend910_95/rs_ub_jetty.c
    ctx/product/ascend910_95/rs_ub_jfc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/dl_hal_function.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/ascend_hal_dl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/file_opt.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/network_comm.c
    $<$<STREQUAL:${PRODUCT_SIDE},device>:${TLV_DIR}/dl_netco_function.c>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:${TLV_DIR}/rs_tlv.c>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:${TLV_DIR}/rs_adp_nslb.c>
)

add_library(rs  SHARED ${rs_list})

target_include_directories(rs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/private/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/
    ${TLV_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    # runtime头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include/

    # 第三方库依赖路径
    ${OPENSSL_INCLUDE_DIR}
    ${CANN_3RD_LIB_PATH}/hcomm_utils/${PRODUCT_SIDE}/include/tls_adp
    ${CANN_3RD_LIB_PATH}/hcomm_utils/${PRODUCT_SIDE}/include/kms

    # rdma头文件
    ${RDMA_CORE_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/host_lite
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/rdma-core/providers/hns
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/nda

    # HCCP Legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/ctx
    ${CMAKE_CURRENT_SOURCE_DIR}/ctx/product
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/net_adapt
    ${URMA_INCLUDE_DIR}
)

if(NOT BUILD_OPEN_PROJECT)
    target_include_directories(rs PRIVATE
        ${TOP_DIR}/ubengine/ssapi/userspace/umdk/src/urma/lib/urma/core/include
        ${TOP_DIR}/ubengine/ssapi/userspace/umdk/src/urma/lib/urma/core
        ${TOP_DIR}/ubengine/ssapi/userspace/udma/src/urma/hw/udma/include
        ${TOP_DIR}/drivers/network/tls_adp
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/driver/driver/pkg_inc/
    )
endif()

# c_sec、mmpa、unified_dlog动态库搜索路径
target_link_directories(rs PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/devlib/device/
)

target_compile_options(rs PRIVATE
    -Werror -fvisibility=hidden -fno-strict-aliasing -fsigned-char -fno-common -Wfloat-equal
    -Wextra -Wno-unused-parameter -std=gnu11
    -fstack-protector-strong
    $<$<CONFIG:Debug>:-g>
)

target_link_options(rs PRIVATE
    -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
    -Wl,-z,now
    $<$<CONFIG:Release>:-s>
)

target_compile_definitions(rs PRIVATE
    LOG_CPP
    $<$<STREQUAL:${PRODUCT_SIDE},host>:LOG_HOST>
    $<$<AND:$<STREQUAL:${USE_ALOG},0>,$<STREQUAL:${PRODUCT_SIDE},device>>:CUSTOM_INTERFACE>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:google=ascend_private>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:CONFIG_TLV>
    CONFIG_V121
)


target_link_libraries(rs PRIVATE
    -Wl,--no-as-needed
    c_sec
    $<$<STREQUAL:${PRODUCT_SIDE},host>:unified_dlog>
    tls_adp
    -Wl,--as-needed
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lrt
    -ldl
)

install(TARGETS  rs
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)

add_dependencies(rs crypto_static)