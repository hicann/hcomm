add_library(topo STATIC control_topo.cpp mesh.cpp mesh_topo.cpp ranktable_info.cpp probe_config.cpp)
target_include_directories(topo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(topo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(topo PUBLIC nlohmann_json)
target_link_libraries(topo PUBLIC file_path)

if(BUILD_TESTS)
    # 启用测试框架支持
    enable_testing()
    # find_package(GTest REQUIRED)
    file(GLOB TOPO_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
    # 为每个测试文件创建单独的可执行文件
    foreach(test_source ${TOPO_TEST_SOURCES})
        # 从文件名获取测试名称（去掉路径和扩展名）
        get_filename_component(test_name ${test_source} NAME_WE)
        # 创建测试可执行文件
        add_executable(${test_name} ${test_source})
        set_target_properties(${test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${DISP_PROBE_INTERNAL_RUNTIME_DIR}")
        # 链接测试所需的库
        target_link_libraries(${test_name} PRIVATE Threads::Threads)
        target_link_libraries(${test_name} PRIVATE topo)
        target_link_libraries(${test_name} PRIVATE CLI)
        # 添加为CTest测试
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    endforeach()
endif()
