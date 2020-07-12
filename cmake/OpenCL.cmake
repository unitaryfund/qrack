set (OPENCL_AMDSDK /opt/AMDAPPSDK-3.0 CACHE PATH "Installation path for the installed AMD OpenCL SDK, if used")

# Options used when building the project
find_package (OpenCL)
if (NOT OpenCL_FOUND)
    # Attempt with AMD's OpenCL SDK
    find_library (LIB_OPENCL OpenCL PATHS ${OPENCL_AMDSDK}/lib/x86_64/)
    if (NOT LIB_OPENCL)
        set (ENABLE_OPENCL OFF)
    else ()
        # Found, set the required include path.
        set (OpenCL_INCLUDE_DIRS ${OPENCL_AMDSDK}/include CACHE PATH "AMD OpenCL SDK Header include path")
        set (OpenCL_COMPILATION_OPTIONS
            -Wno-ignored-attributes
            -Wno-deprecated-declarations
            CACHE STRING "AMD OpenCL SDK Compilation Option Requirements")
        message ("OpenCL support found in the AMD SDK")
    endif()
endif ()

message ("OpenCL Support is: ${ENABLE_OPENCL}")

if (ENABLE_OPENCL)
    target_compile_definitions (qrack PUBLIC CL_HPP_TARGET_OPENCL_VERSION=200)
    target_compile_definitions (qrack PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=110)
    target_compile_definitions (qrack_pinvoke PUBLIC CL_HPP_TARGET_OPENCL_VERSION=200)
    target_compile_definitions (qrack_pinvoke PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=110)

    if (ENABLE_SNUCL)
        find_package(MPI REQUIRED)
        set(QRACK_OpenCL_LIBRARIES snucl_cluster)
        set(QRACK_OpenCL_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH} $ENV{SNUCLROOT}/inc)
        set(QRACK_OpenCL_LINK_DIRS $ENV{SNUCLROOT}/lib)
        set(QRACK_OpenCL_COMPILATION_OPTIONS ${MPI_CXX_COMPILE_FLAGS} ${OpenCL_COMPILATION_OPTIONS} -Wno-deprecated-declarations -Wno-ignored-attributes)

        target_link_directories (qrack_pinvoke PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (unittest PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (benchmarks PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (accuracy PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (qrack_cl_precompile PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (grovers PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (grovers_lookup PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (ordered_list_search PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (quantum_perceptron PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (quantum_associative_memory PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (shors_factoring PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (pearson32 PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (teleport PRIVATE ${QRACK_OpenCL_LINK_DIRS})
        target_link_directories (qneuron_classification PRIVATE ${QRACK_OpenCL_LINK_DIRS})

        target_compile_definitions (qrack PUBLIC ENABLE_SNUCL=1)
    else (ENABLE_SNUCL)
        set(QRACK_OpenCL_LIBRARIES ${OpenCL_LIBRARIES})
        set(QRACK_OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
        set(QRACK_OpenCL_COMPILATION_OPTIONS ${OpenCL_COMPILATION_OPTIONS})

        target_compile_definitions (qrack PUBLIC ENABLE_SNUCL=0)
    endif (ENABLE_SNUCL)

    message ("SnuCL Support is: ${ENABLE_SNUCL}")
    message ("    libOpenCL: ${QRACK_OpenCL_LIBRARIES}")
    message ("    Includes:  ${QRACK_OpenCL_INCLUDE_DIRS}")
    message ("    Options:   ${QRACK_OpenCL_COMPILATION_OPTIONS}")

    target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${QRACK_OpenCL_INCLUDE_DIRS})
    target_compile_options (qrack PUBLIC ${QRACK_OpenCL_COMPILATION_OPTIONS})

    target_link_libraries (qrack_pinvoke ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (unittest ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (benchmarks ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (accuracy ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (qrack_cl_precompile ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (grovers ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (grovers_lookup ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (ordered_list_search ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (quantum_perceptron ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (quantum_associative_memory ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (shors_factoring ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (pearson32 ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (teleport ${QRACK_OpenCL_LIBRARIES})
    target_link_libraries (qneuron_classification ${QRACK_OpenCL_LIBRARIES})

    # Build the OpenCL command files
    find_program (XXD_BIN xxd)
    file (GLOB_RECURSE COMPILABLE_RESOURCES "src/common/*.cl")
    foreach (INPUT_FILE ${COMPILABLE_RESOURCES})
        get_filename_component (INPUT_NAME ${INPUT_FILE} NAME)
        get_filename_component (INPUT_BASENAME ${INPUT_FILE} NAME_WE)
        get_filename_component (INPUT_DIR ${INPUT_FILE} DIRECTORY)

        set (OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/include/common/${INPUT_BASENAME}cl.hpp)

        message (" Creating XXD Rule for ${INPUT_FILE} -> ${OUTPUT_FILE}")
        add_custom_command (
            WORKING_DIRECTORY ${INPUT_DIR}
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${XXD_BIN} -i ${INPUT_NAME} > ${OUTPUT_FILE}
            COMMENT "Building OpenCL Commands in ${INPUT_FILE}"
            )
        list (APPEND COMPILED_RESOURCES ${OUTPUT_FILE})
    endforeach ()

    # Add the OpenCL objects to the library
    target_sources (qrack PRIVATE
        ${COMPILED_RESOURCES}
        src/common/oclengine.cpp
        src/qengine/opencl.cpp
        src/qunitmulti.cpp
        )

else (ENABLE_OPENCL)
    target_compile_definitions (qrack PUBLIC ENABLE_OPENCL=0)
endif (ENABLE_OPENCL)
