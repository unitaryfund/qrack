option (ENABLE_UINT32 "Use 32-bit (instead of 64-bit) unsigned integer types for coherent addressable qubit masks" OFF)
option (ENABLE_UINT128 "128 bit capacity (limited OpenCL support)" OFF)

if (ENABLE_UINT128)
    set(QBCAPPOW "7")
endif (ENABLE_UINT128)
