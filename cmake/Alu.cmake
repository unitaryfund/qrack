option (ENABLE_ALU "Include general ALU API (on by default)" ON)

if (ENABLE_ALU)
    target_sources (qrack PRIVATE
        src/qinterface/arithmetic.cpp
        src/qalu.cpp
        )
endif (ENABLE_ALU)
