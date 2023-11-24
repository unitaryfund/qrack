option (ENABLE_CUDA "Build CUDA-based QEngine type" OFF)

if (ENABLE_OPENCL)
    set(ENABLE_CUDA OFF)
endif ()

if (ENABLE_CUDA)
    find_package (CUDAToolkit)
    if (NOT CUDAToolkit_FOUND)
        set(ENABLE_CUDA OFF)
    endif ()
endif()

message ("CUDA Support is: ${ENABLE_CUDA}")

if (ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(qrack PUBLIC ENABLE_CUDA=1)
    target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options (qrack PUBLIC ${CUDAToolkit_COMPILATION_OPTIONS})

    if (NOT DEFINED QRACK_CUDA_ARCHITECTURES)
        # See https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures#answer-68223399
        if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
            include(FindCUDA/select_compute_arch)
            CUDA_DETECT_INSTALLED_GPUS(QRACK_CUDA_ARCHITECTURES)
            string(STRIP "${QRACK_CUDA_ARCHITECTURES}" QRACK_CUDA_ARCHITECTURES)
            string(REPLACE " " ";" QRACK_CUDA_ARCHITECTURES "${QRACK_CUDA_ARCHITECTURES}")
            string(REPLACE "." "" QRACK_CUDA_ARCHITECTURES "${QRACK_CUDA_ARCHITECTURES}")
        else (${CMAKE_VERSION} VERSION_LESS "3.24.0")
            set(QRACK_CUDA_ARCHITECTURES native)
        endif (${CMAKE_VERSION} VERSION_LESS "3.24.0")
    endif (NOT DEFINED QRACK_CUDA_ARCHITECTURES)
    set(QRACK_CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})

    message("QRACK_CUDA_ARCHITECTURES: ${QRACK_CUDA_ARCHITECTURES}")

    target_link_libraries (qrack ${QRACK_CUDA_LIBRARIES})
    set_target_properties(qrack PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
    target_link_libraries (qrack_pinvoke ${QRACK_CUDA_LIBRARIES})
    set_target_properties(qrack_pinvoke PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
    if (NOT ENABLE_EMIT_LLVM)
        # Declare the OCL precompilation executable
        add_executable (qrack_cl_precompile
            src/qrack_cl_precompile.cpp
            )
        target_link_libraries (qrack_cl_precompile ${QRACK_LIBS})
        target_compile_options (qrack_cl_precompile PUBLIC ${TEST_COMPILE_OPTS})
        install(TARGETS qrack_cl_precompile RUNTIME DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

        if (ENABLE_EXAMPLES)
            target_link_libraries (unittest ${QRACK_CUDA_LIBRARIES})
            set_target_properties(unittest PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
            target_link_libraries (benchmarks ${QRACK_CUDA_LIBRARIES})
            set_target_properties(benchmarks PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
            target_link_libraries (qrack_cl_precompile ${QRACK_CUDA_LIBRARIES})
            set_target_properties(qrack_cl_precompile PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
            target_link_libraries (quantum_associative_memory ${QRACK_CUDA_LIBRARIES})
            set_target_properties(quantum_associative_memory PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
            target_link_libraries (teleport ${QRACK_CUDA_LIBRARIES})
            set_target_properties(teleport PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
            target_link_libraries (qneuron_classification ${QRACK_CUDA_LIBRARIES})
            set_target_properties(qneuron_classification PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
            target_link_libraries (cosmology ${QRACK_CUDA_LIBRARIES})
            set_target_properties(cosmology PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
            if (ENABLE_ALU)
                target_link_libraries (grovers ${QRACK_CUDA_LIBRARIES})
                set_target_properties(grovers PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
                target_link_libraries (grovers_lookup ${QRACK_CUDA_LIBRARIES})
                set_target_properties(grovers_lookup PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
                target_link_libraries (ordered_list_search ${QRACK_CUDA_LIBRARIES})
                set_target_properties(ordered_list_search PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
                target_link_libraries (shors_factoring ${QRACK_CUDA_LIBRARIES})
                set_target_properties(shors_factoring PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
                target_link_libraries (pearson32 ${QRACK_CUDA_LIBRARIES})
                set_target_properties(pearson32 PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
                target_link_libraries (quantum_perceptron ${QRACK_CUDA_LIBRARIES})
                set_target_properties(quantum_perceptron PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
            endif (ENABLE_ALU)
        endif (ENABLE_EXAMPLES)
    endif (NOT ENABLE_EMIT_LLVM)
    
    # Add the CUDA objects to the library
    target_sources (qrack PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/src/common/cudaengine.cu
        ${CMAKE_CURRENT_BINARY_DIR}/src/common/qengine.cu
        ${CMAKE_CURRENT_BINARY_DIR}/src/qengine/cuda.cu
        ${CMAKE_CURRENT_BINARY_DIR}/src/qhybrid.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/src/qunitmulti.cpp
        )

endif(ENABLE_CUDA)
