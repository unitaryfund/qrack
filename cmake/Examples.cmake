add_executable (grovers
    examples/grovers.cpp
    )

set_target_properties(grovers PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

target_link_libraries (grovers
    qrack
    pthread
    )

add_executable (grovers_lookup
    examples/grovers_lookup.cpp
    )

set_target_properties(grovers_lookup PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

target_link_libraries (grovers_lookup
    qrack
    pthread
    )

add_executable (ordered_list_search
    examples/ordered_list_search.cpp
    )

set_target_properties(ordered_list_search PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

target_link_libraries (ordered_list_search
    qrack
    pthread
    )


target_compile_options (grovers PUBLIC -O3 -std=c++11 -Wall -Werror ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (grovers_lookup PUBLIC -O3 -std=c++11 -Wall -Werror ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (ordered_list_search PUBLIC -O3 -std=c++11 -Wall -Werror ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
