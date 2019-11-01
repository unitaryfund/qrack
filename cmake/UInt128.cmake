option (ENABLE_UINT128 "128 bit capacity (not OpenCL compatible)" OFF)

if (ENABLE_UINT128)
    set(ENABLE_OPENCL OFF)
    target_compile_definitions (qrack PUBLIC ENABLE_UINT128=1)
endif (ENABLE_UINT128)
