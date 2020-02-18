cmake_minimum_required (VERSION 3.9)
project (Qrack VERSION 3.1 DESCRIPTION "High Performance Quantum Bit Simulation")

# Installation commands
include (GNUInstallDirs)

include ("cmake/Coverage.cmake")
include ("cmake/Format.cmake")

include_directories ("include" "include/common")

set(PSTRIDE "16" CACHE STRING "Stride of parallel for loops (must be power of 2)")

# Declare the library
add_library (qrack STATIC
    src/common/parallel_for.cpp
    src/common/rdrandwrapper.cpp
    src/qinterface/arithmetic.cpp
    src/qinterface/gates.cpp
    src/qinterface/logic.cpp
    src/qinterface/protected.cpp
    src/qinterface/qinterface.cpp
    src/qinterface/rotational.cpp
    src/qengine/qengine.cpp
    src/qengine/arithmetic.cpp
    src/qengine/gates.cpp
    src/qengine/state.cpp
    src/qengine/utility.cpp
    src/bitbuffer.cpp
    src/pinvoke_api.cpp
    src/qfusion.cpp
    src/qunit.cpp
    )

if (MSVC)
    set(QRACK_LIBS qrack atomic)
else (MSVC)
    set(QRACK_LIBS qrack pthread atomic)
endif (MSVC)

# Declare the unittest executable
add_executable (unittest
    test/test_main.cpp
    test/tests.cpp
    )

target_link_libraries (unittest ${QRACK_LIBS})

add_test (NAME qrack_tests
    COMMAND unittest
    )

# Declare the benchmark executable
add_executable (benchmarks
    test/benchmarks_main.cpp
    test/benchmarks.cpp
    )

target_link_libraries (benchmarks ${QRACK_LIBS})

add_test (NAME qrack_benchmarks
    COMMAND benchmarks
    )

# Declare the accuracy executable
add_executable (accuracy
    test/accuracy_main.cpp
    test/accuracy.cpp
    )

target_link_libraries (accuracy ${QRACK_LIBS})

add_test (NAME qrack_accuracy
    COMMAND accuracy
    )

# Declare the OCL precompilation executable
add_executable (qrack_cl_precompile
    src/qrack_cl_precompile.cpp
    )

target_link_libraries (qrack_cl_precompile ${QRACK_LIBS})

add_test (NAME qrack_cl_precompile
    COMMAND qrack_cl_precompile
    )

# Included after the library and other modules have been declared
option (ENABLE_OPENCL "Use OpenCL optimizations" ON)
include ("cmake/Examples.cmake")
include ("cmake/UInt128.cmake")
include ("cmake/OpenCL.cmake" )
include ("cmake/Complex8.cmake")
include ("cmake/Complex_x2.cmake")
include ("cmake/Pure32.cmake")
include ("cmake/VM6502Q.cmake")

message ("Pure 32-bit compilation is: ${ENABLE_PURE32}")
message ("128-bit compilation is: ${ENABLE_UINT128}")
message ("Single accuracy is: ${ENABLE_COMPLEX8}")
message ("Complex_x2/AVX Support is: ${ENABLE_COMPLEX_X2}")
message ("VM6502Q disassembler support is: ${ENABLE_VM6502Q_DEBUG}")

if (ENABLE_UINT128 AND ENABLE_PURE32)
    message(FATAL_ERROR "You cannot require both ENABLE_UINT128 and ENABLE_PURE32 at the same time! 128 bit and pure 32 bit modes are mutually exclusive.")
endif (ENABLE_UINT128 AND ENABLE_PURE32)

if (MSVC)
    set(QRACK_COMPILE_OPTS -std=c++11 -Wall)
    set(TEST_COMPILE_OPTS -std=c++11 -Wall)
else (MSVC)
    set(QRACK_COMPILE_OPTS -O3 -std=c++11 -Wall -Werror -fPIC)
    set(TEST_COMPILE_OPTS -O3 -std=c++11 -Wall -Werror)
endif(MSVC)

if (ENABLE_COMPLEX_X2 AND NOT ENABLE_COMPLEX8)
    set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -mavx)
endif (ENABLE_COMPLEX_X2 AND NOT ENABLE_COMPLEX8)

include ("cmake/RDRand.cmake")

configure_file(include/common/config.h.in include/common/config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/common)

enable_testing()

# Run the unittest executable on 'make test'

target_include_directories (unittest PUBLIC test)
target_include_directories (benchmarks PUBLIC test)

if (APPLE)
    set(TEST_COMPILE_OPTS -Wno-inconsistent-missing-override)
endif (APPLE)

target_compile_options (qrack PUBLIC ${QRACK_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (unittest PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (benchmarks PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (qrack_cl_precompile PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)

set_target_properties (qrack PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

# Install common headers
install (FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/common/config.h
    include/common/qrack_types.hpp
    include/common/complex16x2simd.hpp
    include/common/complex8x2simd.hpp
    include/common/oclengine.hpp
    include/common/parallel_for.hpp
    include/common/rdrandwrapper.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack/common
    )

# Install qrack library headers
install (FILES
    include/bitbuffer.hpp
    include/hamiltonian.hpp
    include/statevector.hpp
	include/pinvoke_api.hpp
    include/qfactory.hpp
    include/qengine.hpp
    include/qengine_cpu.hpp
    include/qfusion.hpp
    include/qunit.hpp
    include/qengine_opencl.hpp
    include/qinterface.hpp
    include/qneuron.hpp
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
install(TARGETS qrack_cl_precompile DESTINATION bin)
