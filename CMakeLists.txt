cmake_minimum_required (VERSION 3.9)
project (Qrack VERSION 1.0 DESCRIPTION "High Performance Quantum Bit Simulation")

# Installation commands
include (GNUInstallDirs)

include ("cmake/Coverage.cmake")
include ("cmake/Format.cmake" )
include ("cmake/AVX.cmake" )

include_directories ("include" "include/common")

# Declare the library
if (ENABLE_AVX)
    add_library (qrack STATIC
        src/common/complex16simd.cpp
        src/common/complex16x2simd.cpp
        src/common/parallel_for.cpp
        src/qinterface/qinterface.cpp
        src/qengine/qengine.cpp
        src/qengine/operators.cpp
        src/qengine/gates.cpp
        src/qengine/rotational.cpp
        src/qengine/state/state.cpp
        src/qengine/state/operators.cpp
        src/qengine/state/gates.cpp
        src/qunit.cpp
        )
else (ENABLE_AVX)
    add_library (qrack STATIC
        src/common/complex16simd.cpp
        src/common/parallel_for.cpp
        src/qinterface/qinterface.cpp
        src/qengine/qengine.cpp
        src/qengine/operators.cpp
        src/qengine/gates.cpp
        src/qengine/rotational.cpp
        src/qengine/state/state.cpp
        src/qengine/state/operators.cpp
        src/qengine/state/gates.cpp
        src/qunit.cpp
        )
endif (ENABLE_AVX)

# Declare the unittest executable
add_executable (unittest
    test/test_main.cpp
    test/tests.cpp
    )

target_link_libraries (unittest
    qrack
    pthread
    )

add_test (NAME qrack_tests
    COMMAND unittest
    )

# Included after the library and other modules have been declared
include ("cmake/OpenCL.cmake" )

enable_testing()

# Run the unittest executable on 'make test'

target_include_directories (unittest PUBLIC test)

if (ENABLE_AVX)
    target_compile_options (qrack PUBLIC -ggdb -std=c++11 -Wall -Werror -mavx -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_options (unittest PUBLIC -ggdb -std=c++11 -Wall -Werror -mavx -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_definitions (qrack PUBLIC ENABLE_AVX=1)
else (ENABLE_AVX)
    target_compile_options (qrack PUBLIC -ggdb -std=c++11 -Wall -Werror -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_options (unittest PUBLIC -ggdb -std=c++11 -Wall -Werror -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_definitions (qrack PUBLIC ENABLE_AVX=0)
endif (ENABLE_AVX)

set_target_properties (qrack PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

# Install common headers
install (FILES
    include/common/complex16simd.hpp
    include/common/complex16x2simd.hpp
    include/common/oclengine.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack/common
    )

# Install qrack library headers
install (FILES 
    include/qengine_cpu.hpp
    include/qunit.hpp
    include/common/oclengine.hpp
    include/common/complex16simd.hpp
    include/common/complex16x2simd.hpp
    include/common/parallel_for.hpp
    include/qengine_opencl.hpp
    include/qinterface.hpp
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

