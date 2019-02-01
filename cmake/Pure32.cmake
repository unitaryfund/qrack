option (ENABLE_PURE32 "Use only 32-bit types or smaller" OFF)
option (ENABLE_RASPBERRYPI "Build a library version that's safe for the Raspberry Pi 3" OFF)

if (ENABLE_RASPBERRYPI)
    set(ENABLE_PURE32 ON)
    target_compile_definitions(qrack PUBLIC ENABLE_RASPBERRYPI=1)
endif(ENABLE_RASPBERRYPI)

if (ENABLE_PURE32)
    set(ENABLE_COMPLEX_X2 OFF)
    set(ENABLE_COMPLEX8 ON)
    target_compile_definitions (qrack PUBLIC ENABLE_PURE32=1)
endif (ENABLE_PURE32)
