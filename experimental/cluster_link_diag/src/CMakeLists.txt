add_subdirectory(util)

set(DISP_PROBE_PUBLIC_EXECUTABLES
    probe_controller
    probe_topo
    rpc_host)

# 获取当前目录下的所有.cpp和.c文件（不包含子目录）
file(GLOB EXEC_SOURCES 
    "*.cpp"
    "*.c"
)
# 为每个源文件创建一个可执行文件
foreach(source ${EXEC_SOURCES})
    get_filename_component(executable_name ${source} NAME_WE)
    add_executable(${executable_name} ${source})
    if(NOT executable_name IN_LIST DISP_PROBE_PUBLIC_EXECUTABLES)
        set_target_properties(${executable_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${DISP_PROBE_INTERNAL_RUNTIME_DIR}")
    endif()
    target_include_directories(${executable_name} PRIVATE
        "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}"
        "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/eigen3")
    target_link_libraries(${executable_name} PRIVATE Threads::Threads)
    target_link_libraries(${executable_name} PRIVATE rpc)
    target_link_libraries(${executable_name} PRIVATE CLI)
    target_link_libraries(${executable_name} PRIVATE util)
    target_link_libraries(${executable_name} PRIVATE nlohmann_json)
    target_link_libraries(${executable_name} PRIVATE hccn_pingpong_rpc)
endforeach()
