option (ENABLE_RDRAND "Use RDRAND hardware random number generation, if available" ON)

if (ENABLE_RDRAND)
    set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -mrdrnd)
    target_compile_definitions(qrack PUBLIC ENABLE_RDRAND=1)
endif (ENABLE_RDRAND)

message ("Try RDRAND is: ${ENABLE_RDRAND}")
