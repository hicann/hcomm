# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_to_test_list
    main.cc
    hccl_api_base_test.cc
    ut_CcuComponent_test.cc
    ut_CcuConn_test.cc
    ut_ccu_microcode.cc
    ut_ccu_mgr_test.cc
    ./ccu_representation/interface/ut_ccu_interface_assist.cc
    ./ccu_device/ut_ccu_res_batch_allocator.cc
    ./ccu_device/ccu_comp/ccu_channel/ccu_pfe/ut_ccu_pfe_cfg_mgr.cc
)

add_executable(hccl_utest_comms_ccu_features ${src_to_test_list})

#头文件搜索路径
target_include_directories(hccl_utest_comms_ccu_features PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    ${HCOMM_DIR}/pkg_inc/hcomm/ccu
)

target_compile_definitions(hccl_utest_comms_ccu_features PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_comms_ccu_features PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_directories(hccl_utest_comms_ccu_features PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hccl_utest_comms_ccu_features PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archi
    -lgcov
    -lrt
    -ldl
)

add_dependencies(hccl_utest_comms_ccu_features hccl_llt)

enable_testing()
add_test(NAME hccl_utest_comms_ccu_features COMMAND hccl_utest_comms_ccu_features)

add_subdirectory(ccu_representation/reps/sync)
add_subdirectory(ccu_representation/reps/control)
add_subdirectory(ccu_representation/reps/common)
add_subdirectory(ccu_representation/reps/data)
add_subdirectory(ccu_representation/reps/arithmetic)
add_subdirectory(ccu_representation/reps/loop)
add_subdirectory(ccu_representation/interface)
