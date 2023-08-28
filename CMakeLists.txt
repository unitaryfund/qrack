cmake_minimum_required (VERSION 3.9)
project (Qrack VERSION 8.17.8 DESCRIPTION "High Performance Quantum Bit Simulation")

# Installation commands
include (GNUInstallDirs)

include ("cmake/Coverage.cmake")
include ("cmake/Format.cmake")
include ("cmake/EmitLlvm.cmake")
# PSTRIDEPOW relates to parallel for loops:
include ("cmake/QUnit_CPU_Parallel.cmake")
include ("cmake/Pthread.cmake")

include_directories ("include" "include/common")

# Declare the library
add_library (qrack STATIC
    src/common/functions.cpp
    src/common/parallel_for.cpp
    src/qinterface/gates.cpp
    src/qinterface/logic.cpp
    src/qinterface/qinterface.cpp
    src/qinterface/rotational.cpp
    src/qengine/qengine.cpp
    src/qengine/arithmetic.cpp
    src/qengine/state.cpp
    src/qengine/utility.cpp
    src/qengineshard.cpp
    src/qunit.cpp
    src/qunitclifford.cpp
    src/qpager.cpp
    src/qstabilizer.cpp
    src/qstabilizerhybrid.cpp
    src/qcircuit.cpp
    )

if (ENABLE_PTHREAD AND ENABLE_QUNIT_CPU_PARALLEL)
    target_sources (qrack PRIVATE
        src/common/dispatchqueue.cpp
        )
endif (ENABLE_PTHREAD AND ENABLE_QUNIT_CPU_PARALLEL)

if (ENABLE_EMIT_LLVM)
    add_library (qrack_pinvoke OBJECT
        src/pinvoke_api.cpp
        )
else (ENABLE_EMIT_LLVM)
    add_library (qrack_pinvoke SHARED
        src/pinvoke_api.cpp
        )
endif (ENABLE_EMIT_LLVM)

set(QRACK_LIBS qrack)
if (ENABLE_PTHREAD)
    if (ANDROID)
        set(QRACK_LIBS ${QRACK_LIBS} atomic)
    endif(ANDROID)
	if (NOT ANDROID AND NOT MSVC)
	    set(QRACK_LIBS ${QRACK_LIBS} pthread)
	endif (NOT ANDROID AND NOT MSVC)
endif (ENABLE_PTHREAD)

if (FPPOW GREATER 6)
    set(QRACK_LIBS ${QRACK_LIBS} quadmath)
endif (FPPOW GREATER 6)

target_link_libraries (qrack_pinvoke ${QRACK_LIBS})
if (NOT ENABLE_EMIT_LLVM)
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

    # Declare the OCL precompilation executable
    add_executable (qrack_cl_precompile
        src/qrack_cl_precompile.cpp
        )

    target_link_libraries (qrack_cl_precompile ${QRACK_LIBS})
endif (NOT ENABLE_EMIT_LLVM)

# Included after the library and other modules have been declared
option (ENABLE_OPENCL "Use OpenCL optimizations" ON)
option (ENABLE_SNUCL "Use SnuCL framework for clusters" OFF)
include ("cmake/Alu.cmake")
include ("cmake/Bcd.cmake")
include ("cmake/Boost.cmake")
include ("cmake/Complex_x2.cmake")
include ("cmake/CppStd.cmake")
include ("cmake/EnvVars.cmake")
include ("cmake/FpPow.cmake")
include ("cmake/OclMemGuards.cmake")
include ("cmake/Pstridepow.cmake")
include ("cmake/Pure32.cmake")
include ("cmake/QbCapPow.cmake")
include ("cmake/Qbdt.cmake")
include ("cmake/QUnit_CPU_Parallel.cmake")
include ("cmake/RegGates.cmake")
include ("cmake/RotApi.cmake")
include ("cmake/UIntPow.cmake")
include ("cmake/VM6502Q.cmake")
if (NOT ENABLE_EMIT_LLVM)
    include ("cmake/Examples.cmake")
endif (NOT ENABLE_EMIT_LLVM)
include ("cmake/OpenCL.cmake" )
include ("cmake/CUDA.cmake")

message ("C++ standard year is (at least): ${CPP_STD}")
message ("Qubit capacity power is: ${QBCAPPOW}")
message ("Floating-point capacity power is: ${FPPOW}")
message ("UInt capacity power is: ${UINTPOW}")
message ("CPU parallelism is: ${ENABLE_PTHREAD}")
message ("CPU work item stride power is: ${PSTRIDEPOW}")
message ("Parallel QUnit->CPU is: ${ENABLE_QUNIT_CPU_PARALLEL}")
message ("Parallel QBdt is: ${ENABLE_QBDT_CPU_PARALLEL}")
message ("Pure 32-bit compilation is: ${ENABLE_PURE32}")
message ("Complex_x2/AVX support is: ${ENABLE_COMPLEX_X2}")
message ("SSE3.0 support is: ${ENABLE_SSE3}")
message ("OpenCL memory guards are: ${ENABLE_OCL_MEM_GUARDS}")
message ("Quantum Binary decision tree (QBDT) inclusion is: ${ENABLE_QBDT}")
message ("General ALU API inclusion is: ${ENABLE_ALU}")
message ("BCD ALU API inclusion is: ${ENABLE_BCD}")
message ("Extended rotation API inclusion is: ${ENABLE_ROT_API}")
message ("Register-spanning API inclusion is: ${ENABLE_REG_GATES}")
message ("Environment variable usage is: ${ENABLE_ENV_VARS}")
message ("VM6502Q disassembler support is: ${ENABLE_VM6502Q_DEBUG}")
message ("Compiling libraries to LLVM IR is: ${ENABLE_EMIT_LLVM}")

if (QBCAPPOW GREATER 6)
    set(QRACK_CUDA_COMPILE_OPTS -O3 -Xcompiler -fpermissive -Xcompiler -fPIC)
else (QBCAPPOW GREATER 6)
    set(QRACK_CUDA_COMPILE_OPTS -O3 -Werror all-warnings  -Xcompiler -fPIC)
endif (QBCAPPOW GREATER 6)

if (CPP_STD GREATER_EQUAL 20)
    set(QRACK_CPP_STD_OPT -std=c++20)
elseif (CPP_STD GREATER_EQUAL 17)
    set(QRACK_CPP_STD_OPT -std=c++17)
elseif (CPP_STD GREATER_EQUAL 13)
    set(QRACK_CPP_STD_OPT -std=c++14)
else ()
    set(QRACK_CPP_STD_OPT -std=c++11)
endif ()

if (MSVC)
    set(QRACK_COMPILE_OPTS ${QRACK_CPP_STD_OPT} -Wall)
    set(TEST_COMPILE_OPTS ${QRACK_CPP_STD_OPT} -Wall)
else (MSVC)
    if (EMSCRIPTEN)
        message ("Compiling for WebAssembly.")
        if (ENABLE_PTHREAD)
            set(QRACK_COMPILE_OPTS -O3 ${QRACK_CPP_STD_OPT} -pthread -Wall -Werror -fPIC)
        else (ENABLE_PTHREAD)
            set(QRACK_COMPILE_OPTS -O3 ${QRACK_CPP_STD_OPT} -Wall -Werror -fPIC)
        endif (ENABLE_PTHREAD)
        set(TEST_COMPILE_OPTS -O3 ${QRACK_CPP_STD_OPT} -Wall -Werror)
        set_target_properties(qrack PROPERTIES COMPILE_FLAGS "-s USE_BOOST_HEADERS=1" LINK_FLAGS "-s USE_BOOST_HEADERS=1")
    else (EMSCRIPTEN)
        set(QRACK_COMPILE_OPTS -O3 ${QRACK_CPP_STD_OPT} -Wall -Werror -fPIC)
        set(TEST_COMPILE_OPTS -O3 ${QRACK_CPP_STD_OPT} -Wall -Werror)
    endif (EMSCRIPTEN)
endif(MSVC)

if (ENABLE_COMPLEX_X2 AND FPPOW GREATER 5)
    set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -mavx)
endif (ENABLE_COMPLEX_X2 AND FPPOW GREATER 5)

if (ENABLE_SSE3)
    set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -msse3)
endif (ENABLE_SSE3)

if (FPPOW LESS 5)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^arm" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -mfp16-format=ieee)
        set(TEST_COMPILE_OPTS ${TEST_COMPILE_OPTS} -mfp16-format=ieee)
    endif ()
endif (FPPOW LESS 5)

if (EMSCRIPTEN AND ENABLE_COMPLEX_X2)
    set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -msimd128 -msse -msse2)
endif (EMSCRIPTEN AND ENABLE_COMPLEX_X2)

if (ENABLE_EMIT_LLVM)
    set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -emit-llvm)
endif (ENABLE_EMIT_LLVM)

include ("cmake/Random.cmake")

configure_file(include/common/config.h.in include/common/config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/common)

if (NOT ENABLE_EMIT_LLVM)
    enable_testing()
    # Run the unittest executable on 'make test'

    target_include_directories (unittest PUBLIC test)
    target_include_directories (benchmarks PUBLIC test)
endif (NOT ENABLE_EMIT_LLVM)

if (APPLE)
    set(TEST_COMPILE_OPTS -Wno-inconsistent-missing-override)
endif (APPLE)

if (ENABLE_CUDA)
    target_compile_options (qrack PUBLIC
        "$<$<COMPILE_LANGUAGE:C>:${QRACK_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${QRACK_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${QRACK_CUDA_COMPILE_OPTS}>"
        -DCATCH_CONFIG_FAST_COMPILE
        )
    target_compile_options (qrack_pinvoke PUBLIC
        "$<$<COMPILE_LANGUAGE:C>:${QRACK_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${QRACK_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${QRACK_CUDA_COMPILE_OPTS}>"
        -DCATCH_CONFIG_FAST_COMPILE
        )
else (ENABLE_CUDA)
    target_compile_options (qrack PUBLIC ${QRACK_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
   target_compile_options (qrack_pinvoke PUBLIC ${QRACK_COMPILE_OPTS})
endif (ENABLE_CUDA)
target_compile_definitions(qrack_pinvoke PUBLIC -DDLL_EXPORTS)
if (NOT ENABLE_EMIT_LLVM)
    target_compile_options (unittest PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_options (benchmarks PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_options (qrack_cl_precompile PUBLIC ${TEST_COMPILE_OPTS})
endif (NOT ENABLE_EMIT_LLVM)
target_compile_definitions(qrack_pinvoke PUBLIC -DDLL_EXPORTS)
if (NOT ENABLE_EMIT_LLVM)
    target_compile_options (unittest PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_options (benchmarks PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_options (qrack_cl_precompile PUBLIC ${TEST_COMPILE_OPTS})
endif (NOT ENABLE_EMIT_LLVM)

set_target_properties (qrack PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

set_target_properties (qrack_pinvoke PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

# Install common headers
install (FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/common/config.h
    include/common/qrack_types.hpp
    include/common/qrack_macros.hpp
    include/common/complex16x2simd.hpp
    include/common/complex8x2simd.hpp
    include/common/oclapi.hpp
    include/common/oclengine.hpp
    include/common/cudaengine.cuh
    include/common/cuda_kernels.cuh
    include/common/parallel_for.hpp
    include/common/rdrandwrapper.hpp
    include/common/cuda_kernels.cuh
    include/common/dispatchqueue.hpp
    include/common/half.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack/common
    )

# Install qrack library headers
install (FILES
    include/hamiltonian.hpp
    include/statevector.hpp
    include/pinvoke_api.hpp
    include/qcircuit.hpp
    include/qfactory.hpp
    include/qengine.hpp
    include/qengine_cpu.hpp
    include/qubitswapmap.hpp
    include/qengineshard.hpp
    include/mpsshard.hpp
    include/qunit.hpp
    include/qunitclifford.hpp
    include/qunitmulti.hpp
    include/qengine_gpu_util.hpp
    include/qengine_opencl.hpp
    include/qengine_cuda.hpp
    include/qinterface.hpp
    include/qalu.hpp
    include/qparity.hpp
    include/qneuron.hpp
    include/qpager.hpp
    include/qhybrid.hpp
    include/qstabilizer.hpp
    include/qstabilizerhybrid.hpp
    include/qbdt.hpp
    include/qbdt_node.hpp
    include/qbdt_node_interface.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack
    )

# Install the archive
install (TARGETS qrack
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

if (NOT ENABLE_EMIT_LLVM)
    install (TARGETS qrack_pinvoke
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
endif (NOT ENABLE_EMIT_LLVM)

# Install the pkg-config file
configure_file (qrack.pc.in qrack.pc @ONLY)
install (FILES ${CMAKE_BINARY_DIR}/qrack.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
if (NOT ENABLE_EMIT_LLVM)
    install(TARGETS qrack_cl_precompile DESTINATION bin)
endif (NOT ENABLE_EMIT_LLVM)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_PACKAGE_VERSION_MAJOR "${Qrack_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Qrack_VERSION_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)
