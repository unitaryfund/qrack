option (ENABLE_CUDA "Build CUDA-based QEngine type" OFF)

if (ENABLE_OPENCL)
    set(ENABLE_CUDA OFF)
endif ()

if (ENABLE_CUDA)
    if ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        find_package(CUDA)
        if (NOT CUDA_FOUND)
            set(ENABLE_CUDA OFF)
        endif ()
    else ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        find_package (CUDAToolkit)
        if (NOT CUDAToolkit_FOUND)
            set(ENABLE_CUDA OFF)
        endif ()
    endif ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
endif()

message ("CUDA Support is: ${ENABLE_CUDA}")

if (ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(qrack PUBLIC ENABLE_CUDA=1)
    if ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${CUDA_INCLUDE_DIRS})
        target_compile_options (qrack PUBLIC ${CUDA_COMPILATION_OPTIONS})
        set(QRACK_CUDA_LIBRARIES ${CUDA_LIBRARIES})
    else ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${CUDAToolkit_INCLUDE_DIRS})
        target_compile_options (qrack PUBLIC ${CUDAToolkit_COMPILATION_OPTIONS})
        set(QRACK_CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})
    endif ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))

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

    message("QRACK_CUDA_ARCHITECTURES: ${QRACK_CUDA_ARCHITECTURES}")

    target_link_libraries (qrack ${QRACK_CUDA_LIBRARIES})
    set_target_properties(qrack PROPERTIES CUDA_ARCHITECTURES ${QRACK_CUDA_ARCHITECTURES})
    
    # Add the CUDA objects to the library
    target_sources (qrack PRIVATE
        src/common/cudaengine.cu
        src/common/qengine.cu
        src/qengine/cuda.cu
        src/qhybrid.cpp
        src/qunitmulti.cpp
        )

endif(ENABLE_CUDA)
