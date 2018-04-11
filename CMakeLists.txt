cmake_minimum_required (VERSION 3.9)
project (QRack VERSION 1.0 DESCRIPTION "High Performance Quantum Bit Simulation")

# Installation commands
include (GNUInstallDirs)

set (OPENCL_AMDSDK /opt/AMDAPPSDK-3.0 CACHE PATH "Installation path for the installed AMD OpenCL SDK, if used")
option (USE_OPENCL "Use OpenCL optimizations" ON)

# Options used when building the project
find_library (LIB_OPENCL OpenCL)
if (NOT LIB_OPENCL)
    # Attempt with AMD's OpenCL SDK
    find_library (LIB_OPENCL OpenCL PATHS ${OPENCL_AMDSDK}/lib/x86_64/)
    if (NOT LIB_OPENCL)
        set (USE_OPENCL OFF)
    else ()
        # Found, set the required include path.
        set (OPENCL_INCLUDE_PATH ${OPENCL_AMDSDK}/include CACHE PATH "AMD OpenCL SDK Header include path")
        set (OPENCL_COMPILATION_OPTIONS
            -Wno-ignored-attributes
            -Wno-deprecated-declarations
            CACHE STRING "AMD OpenCL SDK Compilation Option Requirements")
        message ("OpenCL support found in the AMD SDK")
    endif ()
endif ()

message ("OpenCL Support is: ${USE_OPENCL}")

if (USE_OPENCL)
    message ("    libOpenCL: ${LIB_OPENCL}")
    message ("    Includes:  ${OPENCL_INCLUDE_PATH}")
    message ("    Options:   ${OPENCL_COMPILATION_OPTIONS}")
endif ()

include_directories ("include" "include/common")

# Declare the library
add_library (qrack STATIC
    src/qregister.cpp
    src/qregister_factory.cpp
    src/qregister_software.cpp
    src/separatedunit.cpp
    src/common/complex16simd.cpp
    )

add_executable (unittest
        test/test_main.cpp
        test/tests.cpp
        )

target_link_libraries (unittest
        qrack
        pthread
        )

target_include_directories (unittest PUBLIC test)

if (USE_OPENCL)
    target_compile_definitions (qrack PUBLIC ENABLE_OPENCL=1)

    # Include the necessary options and libraries to link against
    target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${OPENCL_INCLUDE_PATH})
    target_compile_options (qrack PUBLIC ${OPENCL_COMPILATION_OPTIONS})
    target_link_libraries (unittest ${LIB_OPENCL})

    # Build the OpenCL command files
    find_program (XXD_BIN xxd)
    file (GLOB_RECURSE COMPILED_RESOURCES "src/*.cl")
    foreach (INPUT_FILE ${COMPILED_RESOURCES})
        get_filename_component (INPUT_NAME ${INPUT_FILE} NAME)
        get_filename_component (INPUT_BASENAME ${INPUT_FILE} NAME_WE)
        get_filename_component (INPUT_DIR ${INPUT_FILE} DIRECTORY)

        set (OUTPUT_FILE ${PROJECT_BINARY_DIR}/${INPUT_BASENAME}cl.hpp)

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
        src/qregister_opencl.cpp
        )

    install (FILES
        include/common/oclengine.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack/common
        )
else (USE_OPENCL)
    target_compile_definitions (qrack PUBLIC ENABLE_OPENCL=0)
endif (USE_OPENCL)

set_target_properties (qrack PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

# Install common headers
install (FILES
    include/common/complex16simd.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack/common
    )

# Install qrack library headers
install (FILES 
    include/qregister.hpp
    include/qregister_factory.hpp
    include/qregister_opencl.hpp
    include/separatedunit.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack
    )

# Install the archive
install (TARGETS qrack
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

# Install the pkg-config file
configure_file (qrack.pc.in qrack.pc @ONLY)
install (FILES ${CMAKE_BINARY_DIR}/qrack.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

