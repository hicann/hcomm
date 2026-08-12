add_library(util INTERFACE)
target_include_directories(util INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(file_path)
add_subdirectory(tool)
add_subdirectory(topo)
add_subdirectory(helper)

target_link_libraries(util INTERFACE file_path)
target_link_libraries(util INTERFACE tool)
target_link_libraries(util INTERFACE topo)
target_link_libraries(util INTERFACE helper)
