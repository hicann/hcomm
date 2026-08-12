add_library(file_path STATIC workspace.cpp)
target_include_directories(file_path PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})