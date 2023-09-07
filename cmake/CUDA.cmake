option (ENABLE_CUDA "Build CUDA-based QEngine type" OFF)

find_package (CUDA)
if (NOT CUDA_FOUND OR ENABLE_OPENCL)
    set(ENABLE_CUDA OFF)
endif ()

message ("CUDA Support is: ${ENABLE_CUDA}")

if (ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(qrack PUBLIC ENABLE_CUDA=1)
    target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${CUDA_INCLUDE_DIRS})
    target_compile_options (qrack PUBLIC ${CUDA_COMPILATION_OPTIONS})

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
    set(QRACK_CUDA_LIBRARIES ${CUDA_LIBRARIES} cutensornet cutensor)

    message("QRACK_CUDA_ARCHITECTURES: ${QRACK_CUDA_ARCHITECTURES}")

    target_link_libraries (qrack ${QRACK_CUDA_LIBRARIES})
    set_target_properties(qrack PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
    target_link_libraries (qrack_pinvoke ${QRACK_CUDA_LIBRARIES})
    set_target_properties(qrack_pinvoke PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
    if (NOT ENABLE_EMIT_LLVM)
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
    endif (NOT ENABLE_EMIT_LLVM)
    
    # Add the CUDA objects to the library
    target_sources (qrack PRIVATE
        src/common/cudaengine.cu
        src/common/qengine.cu
        src/qengine/cuda.cu
        src/qhybrid.cpp
        src/qunitmulti.cpp
        )
endif(ENABLE_CUDA)
