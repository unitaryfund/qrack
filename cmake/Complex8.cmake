option (ENABLE_COMPLEX8 "Use 32 bit float accuracy instead of 64 bit float accuracy")
message ("Single accuracy is: ${ENABLE_COMPLEX8}")

if (ENABLE_COMPLEX8)
    target_compile_definitions (qrack PUBLIC ENABLE_COMPLEX8=1)
endif (ENABLE_COMPLEX8)
