option (ENABLE_PURE32 "Use only 32-bit types or smaller" OFF)
option (ENABLE_VC4CL "Build a library version that's safe for the VC4CL compiler (for the Raspberry Pi 3)" OFF)

if (ENABLE_VC4CL)
    set(ENABLE_PURE32 ON)
    target_compile_definitions(qrack PUBLIC ENABLE_VC4CL=1)
endif(ENABLE_VC4CL)

if (ENABLE_PURE32)
    set(ENABLE_COMPLEX_X2 OFF)
    set(FPPOW "5")
    set(QBCAPPOW "5")
endif (ENABLE_PURE32)
