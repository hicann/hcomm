# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

unset(openssl_FOUND CACHE)
unset(CRYPTO_INCLUDE CACHE)
unset(SSL_INCLUDE CACHE)
unset(CRYPTO_STATIC_LIBRARY CACHE)
unset(SSL_STATIC_LIBRARY CACHE)

if(CCACHE_PROGRAM)
    set(OPENSSL_CC "${CCACHE_PROGRAM} ${CMAKE_C_COMPILER}")
    set(OPENSSL_CXX "${CCACHE_PROGRAM} ${CMAKE_CXX_COMPILER}")
else()
    set(OPENSSL_CC "${CMAKE_C_COMPILER}")
    set(OPENSSL_CXX "${CMAKE_CXX_COMPILER}")
endif()

set(OPENSSL_FILE "openssl-openssl-3.0.9.tar.gz")
set(OPENSSL_URL "https://gitcode.com/cann-src-third-party/openssl/releases/download/openssl-3.0.9/${OPENSSL_FILE}")
set(OPENSSL_PKG_PATH ${CANN_3RD_LIB_PATH}/${OPENSSL_FILE})
set(OPENSSL_INSTALL_PATH ${CANN_3RD_LIB_PATH}/openssl-${PRODUCT_SIDE})
set(OPENSSL_SRC_PATH ${PROJECT_SOURCE_DIR}/openssl-${PRODUCT_SIDE}-src)
set(OPENSSL_INCLUDE_DIR
    ${OPENSSL_INSTALL_PATH}/include
    ${OPENSSL_SRC_PATH}/include
)

# 查找目录下是否已经安装，避免重复编译安装
message(STATUS "[ThirdParty] OPENSSL_INSTALL_PATH=${OPENSSL_INCLUDE_DIR}")
find_path(CRYPTO_INCLUDE
    NAMES crypto/x509.h
    PATH_SUFFIXES include
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${OPENSSL_INCLUDE_DIR}
)
find_path(SSL_INCLUDE
    NAMES openssl/ssl.h
    PATH_SUFFIXES include
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${OPENSSL_INSTALL_PATH}
)
find_library(CRYPTO_STATIC_LIBRARY
    NAMES libcrypto.a
    PATH_SUFFIXES lib lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${OPENSSL_INSTALL_PATH}
)
find_library(SSL_STATIC_LIBRARY
    NAMES libssl.a
    PATH_SUFFIXES lib lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${OPENSSL_INSTALL_PATH}
)

# 是否找到 openssl 的库文件
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(openssl
    FOUND_VAR
    openssl_FOUND 
    REQUIRED_VARS
    CRYPTO_INCLUDE
    SSL_INCLUDE
    CRYPTO_STATIC_LIBRARY
    SSL_STATIC_LIBRARY
)
message(STATUS "[ThirdParty] Found openssl: ${openssl_FOUND}")

if(openssl_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[ThirdParty] openssl found in ${OPENSSL_INSTALL_PATH}, and not force rebuild cann third_party")
else()
    if(EXISTS ${OPENSSL_PKG_PATH})
        # 离线编译场景，优先使用已下载的包
        message(STATUS "[ThirdParty] Found local openssl package: ${OPENSSL_PKG_PATH}")
        set(OPENSSL_PROJECT_URL ${OPENSSL_PKG_PATH})
    elseif(EXISTS ${CANN_3RD_LIB_PATH}/openssl)
        # 离线编译场景，优先使用源代码目录
        file(COPY ${CANN_3RD_LIB_PATH}/openssl/ DESTINATION ${OPENSSL_SRC_PATH})
        message(STATUS "[ThirdParty] Found local openssl source: ${CANN_3RD_LIB_PATH}/openssl")
        set(OPENSSL_PROJECT_URL "")
    else()
        # 下载并编译安装
        message(STATUS "[ThirdParty] Downloading openssl from ${OPENSSL_URL}")
        set(OPENSSL_PROJECT_URL ${OPENSSL_URL})
    endif()

    # ========== 工具链配置（根据系统架构判断） ==========
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(OPENSSL_PLATFORM linux-x86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(OPENSSL_PLATFORM linux-aarch64)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
        set(OPENSSL_PLATFORM linux-armv4)
    else()
        set(OPENSSL_PLATFORM linux-generic64)
    endif()

    # ========== 编译选项 ==========
    set(OPENSSL_OPTION "-fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 -Wl,-z,relro,-z,now,-z,noexecstack -Wl,--build-id=none -s")

    if("${TOOLCHAIN_DIR}" STREQUAL "arm-tiny-hcc-toolchain.cmake")
        set(OPENSSL_OPTION "-mcpu=cortex-a55 -mfloat-abi=hard ${OPENSSL_OPTION}")
    elseif("${TOOLCHAIN_DIR}" STREQUAL "arm-nano-hcc-toolchain.cmake")
        set(OPENSSL_OPTION "-mcpu=cortex-a9 -mfloat-abi=soft ${OPENSSL_OPTION}")
    endif()

    # ========== Perl 路径（OpenSSL 的 configure 依赖 Perl）==========
    find_program(PERL_PATH perl REQUIRED)

    set(OPENSSL_CONFIGURE_PUB_COMMAND
        ${PERL_PATH} <SOURCE_DIR>/Configure
        ${OPENSSL_PLATFORM}
        no-asm no-shared threads enable-ssl3-method no-tests
        ${OPENSSL_OPTION}
        --prefix=${OPENSSL_INSTALL_PATH}
    )

    if(DEVICE_MODE)
        set(OPENSSL_CONFIGURE_COMMAND
            unset CROSS_COMPILE &&
            ${OPENSSL_CONFIGURE_PUB_COMMAND}
        )
        set(OPENSSL_INSTALL_LIBPATH lib)
    else()
        set(OPENSSL_CONFIGURE_COMMAND
            unset CROSS_COMPILE &&
            export NO_OSSL_RENAME_VERSION=1 &&
            ${OPENSSL_CONFIGURE_PUB_COMMAND}
        )
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") 
            set(OPENSSL_INSTALL_LIBPATH lib64) 
        else() 
            set(OPENSSL_INSTALL_LIBPATH lib) 
        endif()
    endif()

    include(ExternalProject)
    ExternalProject_Add(third_party_openssl
        URL ${OPENSSL_PROJECT_URL}
        URL_HASH SHA256=2eec31f2ac0e126ff68d8107891ef534159c4fcfb095365d4cd4dc57d82616ee
        TLS_VERIFY OFF
        DOWNLOAD_NO_PROGRESS TRUE
        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}
        SOURCE_DIR ${OPENSSL_SRC_PATH}                # 解压后的源码目录
        CONFIGURE_COMMAND ${OPENSSL_CONFIGURE_COMMAND} CC=${OPENSSL_CC} CXX={OPENSSL_CXX}
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install_dev
        BUILD_IN_SOURCE TRUE                          # OpenSSL 不支持分离构建目录
    )
endif()

# 创建导入的目标
add_library(crypto_static STATIC IMPORTED GLOBAL)
add_library(ssl_static STATIC IMPORTED GLOBAL)
add_dependencies(crypto_static ssl_static third_party_openssl)
set_target_properties(crypto_static PROPERTIES
    IMPORTED_LOCATION "${OPENSSL_INSTALL_PATH}/${OPENSSL_INSTALL_LIBPATH}/libcrypto.a"
)
set_target_properties(ssl_static PROPERTIES
    IMPORTED_LOCATION "${OPENSSL_INSTALL_PATH}/${OPENSSL_INSTALL_LIBPATH}/libssl.a"
)
