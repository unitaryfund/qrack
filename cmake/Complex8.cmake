option (ENABLE_COMPLEX8 "Enable complex number float accuracy, over double")
message ("Single accuracy is: ${ENABLE_COMPLEX8}")

if (ENABLE_COMPLEX8)
    target_compile_definitions (qrack PUBLIC ENABLE_COMPLEX8=1)
endif (ENABLE_COMPLEX8)
