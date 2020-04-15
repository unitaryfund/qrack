option (ENABLE_UINT128 "128 bit capacity (limited OpenCL support)" OFF)

if (ENABLE_UINT128)
    target_compile_definitions (qrack PUBLIC ENABLE_UINT128=1)
endif (ENABLE_UINT128)
