option (BUILD_SHARED_LIBS "Build shared libraries" ON)
option (BUILD_DLL "Build DLL for Q# support" ON)

if (BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
    target_compile_definitions(qrack_pinvoke PUBLIC BUILD_SHARED_LIBS=1)
endif(BUILD_SHARED_LIBS)

if (BUILD_DLL)
    set(BUILD_DLL ON)
    target_compile_definitions (qrack_pinvoke PUBLIC BUILD_DLL=1)
endif (BUILD_DLL)

