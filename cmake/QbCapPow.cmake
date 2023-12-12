set(QBCAPPOW "7" CACHE STRING "Log2 of maximum qubit capacity of a single QInterface (must be at least 5, equivalent to >= 32 qubits)")

if (QBCAPPOW LESS 5)
    message(FATAL_ERROR "QBCAPPOW must be at least 5, equivalent to >= 32 qubits!")
endif (QBCAPPOW LESS 5)

if (QBCAPPOW GREATER 6)
    target_sources(qrack PRIVATE
        src/common/big_integer.cpp
        )
endif (QBCAPPOW GREATER 6)
