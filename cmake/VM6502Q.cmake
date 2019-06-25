option (ENABLE_VM6502Q_DEBUG "Build so that the VM6502Q disassembler accurately follows Ehrenfest's theorem" OFF)
if (ENABLE_VM6502Q_DEBUG)
    target_compile_definitions (qrack PUBLIC ENABLE_VM6502Q_DEBUG=1)
endif (ENABLE_VM6502Q_DEBUG)
