set(UINTPOW "6" CACHE STRING "Log2 of \"local qubit\" capacity in QEngine types (not QPager or QUnit, must be at least 4, equivalent to short masks)")
if (UINTPOW LESS 3)
    message(FATAL_ERROR "UINTPOW must be at least 3, equivalent to \"unsigned char\" masks!")
endif (UINTPOW LESS 3)

if (UINTPOW GREATER 6)
    message(FATAL_ERROR "UINTPOW must be no greater than 6, equivalent to \"long\" masks!")
endif (UINTPOW GREATER 6)

math(EXPR _UINTCAP "1 << ${UINTPOW}")
if (NOT (_UINTCAP GREATER PSTRIDEPOW))
    message(FATAL_ERROR "PSTRIDEPOW must be less than 2-to-the-power-of-UINTPOW, or parallel loops will truncate out of bounds!")
endif (NOT (_UINTCAP GREATER PSTRIDEPOW))
