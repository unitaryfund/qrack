option (ENABLE_RDRAND "Use RDRAND hardware random number generation, if available" ON)

if (ENABLE_CUDA)
    set(ENABLE_RDRAND OFF)
endif (ENABLE_CUDA)

if (ENABLE_RDRAND)
    set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -mrdrnd)
    target_compile_definitions(qrack PUBLIC ENABLE_RDRAND=1)
endif (ENABLE_RDRAND)

message ("Try RDRAND is: ${ENABLE_RDRAND}")

option (ENABLE_RNDFILE "Get random numbers from ~/.qrack/rng directory" OFF)

if (ENABLE_RNDFILE)
    target_compile_definitions(qrack PUBLIC ENABLE_RNDFILE=1)
endif (ENABLE_RNDFILE)

message ("RNDFILE is: ${ENABLE_RNDFILE}")
