add_library(tool STATIC 
    cmd/get_IP.cpp 
    tracert/tracert.cpp 
    metrics_collector/metrics_collector.cpp)
target_include_directories(tool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(tool PUBLIC nlohmann_json)
target_link_libraries(tool PUBLIC rpc)


# find_package(GTest REQUIRED)
file(GLOB TOOL_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*/*_test.cpp)
# 为每个测试文件创建单独的可执行文件
foreach(test_source ${TOOL_TEST_SOURCES})
    # 从文件名获取测试名称（去掉路径和扩展名）
    get_filename_component(test_name ${test_source} NAME_WE)
    # 创建测试可执行文件
    add_executable(${test_name} ${test_source})
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${DISP_PROBE_INTERNAL_RUNTIME_DIR}")
    # 链接测试所需的库（包括主项目库）
    target_link_libraries(${test_name} PRIVATE Threads::Threads)
    target_link_libraries(${test_name} PRIVATE tool)
    target_link_libraries(${test_name} PRIVATE CLI)
    # 添加为CTest测试
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()
add_subdirectory(hccn_pingpong)
