add_executable (grovers
    examples/grovers.cpp
    )

set_target_properties(grovers PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

target_link_libraries (grovers ${QRACK_LIBS})

add_executable (grovers_lookup
    examples/grovers_lookup.cpp
    )

set_target_properties(grovers_lookup PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

target_link_libraries (grovers_lookup ${QRACK_LIBS})

add_executable (ordered_list_search
    examples/ordered_list_search.cpp
    )

set_target_properties(ordered_list_search PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

target_link_libraries (ordered_list_search ${QRACK_LIBS})

add_executable (quantum_perceptron
    examples/quantum_perceptron.cpp
    )

set_target_properties(quantum_perceptron PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

target_link_libraries (quantum_perceptron ${QRACK_LIBS})

target_compile_options (grovers PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (grovers_lookup PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (ordered_list_search PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (quantum_perceptron PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
