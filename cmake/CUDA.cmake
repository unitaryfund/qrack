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
    
    # Add the CUDA objects to the library
    target_sources (qrack PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/src/common/cudaengine.cu
        ${CMAKE_CURRENT_BINARY_DIR}/src/common/qengine.cu
        ${CMAKE_CURRENT_BINARY_DIR}/src/qengine/cuda.cu
        ${CMAKE_CURRENT_BINARY_DIR}/src/qhybrid.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/src/qunitmulti.cpp
        )

endif(ENABLE_CUDA)
