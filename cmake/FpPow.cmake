set(FPPOW "5" CACHE STRING "Log2 of float bits, for use in pairs as complex amplitudes (must be at least 2, equivalent to half precision)")
if (FPPOW LESS 4)
    message(FATAL_ERROR "FPPOW must be at least 4, equivalent to \"half\" precision!")
endif (FPPOW LESS 4)

if (FPPOW GREATER 6)
    message(FATAL_ERROR "FPPOW must be no greater than 6, equivalent to \"double\" precision!")
endif (FPPOW GREATER 6)
