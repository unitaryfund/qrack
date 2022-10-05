option (ENABLE_PTHREAD "Enable pthread parallelism (for CPU)" ON)
if (NOT ENABLE_PTHREAD)
    set(ENABLE_QUNIT_CPU_PARALLEL OFF)
endif (NOT ENABLE_PTHREAD)
