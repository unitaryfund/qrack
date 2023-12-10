set(QBCAPPOW "7" CACHE STRING "Log2 of maximum qubit capacity of a single QInterface (must be at least 6, equivalent to >= 64 qubits)")
if (QBCAPPOW LESS 6)
    message(FATAL_ERROR "QBCAPPOW must be at least 6, equivalent to >= 64 qubits!")
endif (QBCAPPOW LESS 6)
