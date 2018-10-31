cmake_minimum_required (VERSION 3.9)
project (Qrack VERSION 1.0 DESCRIPTION "High Performance Quantum Bit Simulation")

# Installation commands
include (GNUInstallDirs)

include ("cmake/Coverage.cmake")
include ("cmake/Format.cmake")

include_directories ("include" "include/common")

set(PSTRIDE "16" CACHE STRING "Stride of parallel for loops")

# Declare the library
add_library (qrack STATIC
    src/common/parallel_for.cpp
    src/qinterface/qinterface.cpp
    src/qinterface/protected.cpp
    src/qinterface/operators.cpp
    src/qinterface/gates.cpp
    src/qinterface/rotational.cpp
    src/qengine/qengine.cpp
    src/qengine/operators.cpp
    src/qengine/gates.cpp
    src/qengine/state.cpp
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
include ("cmake/Complex8.cmake")
include ("cmake/Complex_x2.cmake")

if (ENABLE_COMPLEX_X2 AND NOT ENABLE_COMPLEX8)
    set(QRACK_COMPILE_OPTS -mavx)
endif (ENABLE_COMPLEX_X2 AND NOT ENABLE_COMPLEX8)

configure_file(include/config.h.in include/config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

enable_testing()

# Run the unittest executable on 'make test'

target_include_directories (unittest PUBLIC test)
target_include_directories (benchmarks PUBLIC test)

if (APPLE)
    set(TEST_COMPILE_OPTS -Wno-inconsistent-missing-override)
endif (APPLE)

target_compile_options (qrack PUBLIC -O3 -std=c++11 -Wall -Werror ${QRACK_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (unittest PUBLIC -O3 -std=c++11 -Wall -Werror ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (benchmarks PUBLIC -O3 -std=c++11 -Wall -Werror ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)

set_target_properties (qrack PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

# Install common headers
install (FILES
    include/common/complex16simd.hpp
    include/common/complex16x2simd.hpp
    include/common/complex8x2simd.hpp
    include/common/oclengine.hpp
    include/common/parallel_for.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack/common
    )

# Install qrack library headers
install (FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.h
    include/qfactory.hpp
    include/qengine.hpp
    include/qengine_cpu.hpp
    include/qunit.hpp
    include/qunitmulti.hpp
    include/qengine_opencl.hpp
    include/qengine_opencl_multi.hpp
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
