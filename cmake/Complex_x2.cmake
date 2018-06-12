option (ENABLE_COMPLEX_X2 "Use Complex type vector optimizations (including AVX)" ON)
message ("Complex_x2/AVX Support is: ${ENABLE_COMPLEX_X2}")

if (ENABLE_COMPLEX_X2)
    target_compile_definitions (qrack PUBLIC ENABLE_COMPLEX_X2=1)
endif (ENABLE_COMPLEX_X2)
