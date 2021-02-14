set(QBCAPPOW "6" CACHE STRING "Log2 of maximum qubit capacity of a single QInterface (must be at least 5, equivalent to >= 32 qubits)")
if (QBCAPPOW LESS 5)
    message(FATAL_ERROR "QBCAPPOW must be at least 5, equivalent to >= 32 qubits!")
endif (QBCAPPOW LESS 5)

if (QBCAPPOW GREATER 31)
    message(FATAL_ERROR "QBCAPPOW must be less than 32, equivalent to < 2^32 qubits!")
endif (QBCAPPOW GREATER 31)
