cmake_minimum_required (VERSION 3.9)
project (Qrack VERSION 1.0 DESCRIPTION "High Performance Quantum Bit Simulation")

# Installation commands
include (GNUInstallDirs)

include ("cmake/Coverage.cmake")
include ("cmake/Format.cmake" )
include ("cmake/AVX.cmake" )

include_directories ("include" "include/common")

set(PSTRIDE "16" CACHE STRING "Stride of parallel for loops")

# Declare the library
add_library (qrack STATIC
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

# Declare the benchmark executable
add_executable (benchmarks
    test/test_main.cpp
    test/benchmarks.cpp
    )

target_link_libraries (benchmarks
    qrack
    pthread
    )

add_test (NAME qrack_benchmarks
    COMMAND benchmarks
    )

# Included after the library and other modules have been declared
include ("cmake/OpenCL.cmake" )

enable_testing()

# Run the unittest executable on 'make test'

target_include_directories (unittest PUBLIC test)
target_include_directories (benchmarks PUBLIC test)

if (ENABLE_COMPLEX8)
    target_compile_definitions (qrack PUBLIC ENABLE_COMPLEX8=1)
endif (ENABLE_COMPLEX8)
if (ENABLE_AVX)
    target_compile_definitions (qrack PUBLIC ENABLE_AVX=1)
    set(QRACK_COMPILE_OPTS -mavx)
endif (ENABLE_AVX)

if (APPLE)
    set(TEST_COMPILE_OPTS -Wno-inconsistent-missing-override)
endif (APPLE)

target_compile_options (qrack PUBLIC -O3 -std=c++11 -Wall -Werror ${QRACK_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE -DPSTRIDE=${PSTRIDE})
target_compile_options (unittest PUBLIC -O3 -std=c++11 -Wall -Werror ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (benchmarks PUBLIC -O3 -std=c++11 -Wall -Werror ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)

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

