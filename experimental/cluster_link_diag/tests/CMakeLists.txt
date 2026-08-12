set(DISP_PROBE_UTS
    t1_config_test
    t2_topology_test
    t3_probe_plan_test
    t4_metric_reduction_test
    t5_link_solver_test
    t6_metrics_collector_test)

foreach(test_name IN LISTS DISP_PROBE_UTS)
    add_executable(${test_name} ${test_name}.cpp)
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${DISP_PROBE_INTERNAL_RUNTIME_DIR}/tests")
    target_include_directories(${test_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/src/util")
    target_link_libraries(${test_name} PRIVATE
        Threads::Threads
        helper
        topo
        tool
        nlohmann_json)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    set_tests_properties(${test_name} PROPERTIES LABELS "UT")
endforeach()
