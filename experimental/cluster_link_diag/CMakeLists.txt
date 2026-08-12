cmake_minimum_required(VERSION 3.10)
project(dist_probe C CXX)

set(THIRDLIB_ROOT "$ENV{THIRDLIB_ROOT}" CACHE PATH "Third-party dependency install prefix")
if(THIRDLIB_ROOT STREQUAL "")
    set(THIRDLIB_ROOT "/usr/local/third_lib" CACHE PATH "Third-party dependency install prefix" FORCE)
endif()
message(STATUS "Third-party dependency root: ${THIRDLIB_ROOT}")


# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# 设置test编译标志
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# DEBUG 配置
option(BUILD_TESTS "Build test executables" OFF)
option(ENABLE_SANITIZER "Enable Address/Undefined sanitizers" OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        -O3
    )
else()
    if(ENABLE_SANITIZER)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            message(STATUS "Sanitizers enabled (Address + Undefined)")

            # 确保使用 Debug 构建类型
            if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
                message(WARNING "Sanitizers work best with Debug build type")
            endif()

            # 编译选项
            add_compile_options(
                -g3 # 包含调试信息
                -O0 # 不优化方便调试
                -fno-omit-frame-pointer
                -fsanitize=address,undefined
                -fno-sanitize-recover=all
            )

            # 链接选项
            add_link_options(-fsanitize=address,undefined)

            # Clang 额外优化
            if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                add_compile_options(
                    -fsanitize-address-use-after-scope
                    -fsanitize=pointer-compare
                    -fsanitize=pointer-subtract
                )
            endif()
        else()
            message(WARNING "Sanitizer not supported for ${CMAKE_CXX_COMPILER_ID}")
        endif()
    endif()
endif()


# 正式入口输出到 build/，测试和辅助可执行文件输出到 build/internal/
set(DISP_PROBE_PUBLIC_RUNTIME_DIR ${CMAKE_BINARY_DIR})
set(DISP_PROBE_INTERNAL_RUNTIME_DIR ${CMAKE_BINARY_DIR}/internal)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DISP_PROBE_PUBLIC_RUNTIME_DIR})

include(cmake/ThirdParty.cmake)
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
